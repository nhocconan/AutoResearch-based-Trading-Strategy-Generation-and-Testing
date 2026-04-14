#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week RSI divergence + volume confirmation + trend filter
# Targets: 20-50 trades/year by requiring multiple confluence factors
# Logic: Long when weekly RSI < 30 (oversold) and price touches 12h VWAP with volume spike
#        Short when weekly RSI > 70 (overbought) and price touches 12h VWAP with volume spike
#        Uses 12h EMA50 as trend filter to avoid counter-trend trades
# Position size: 0.25 to manage drawdown in volatile markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly RSI (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Calculate 12h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / np.where(vwap_denominator > 0, vwap_denominator, np.nan)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average (20) for volume spike detection
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned weekly RSI
        rsi_1w_i = align_htf_to_ltf(prices, df_1w, rsi_1w)[i]
        
        if np.isnan(rsi_1w_i) or np.isnan(vwap[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # Volume spike (2x average volume)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Price near VWAP (within 0.5%)
        price_to_vwap_ratio = close[i] / vwap[i]
        near_vwap = (price_to_vwap_ratio > 0.995) and (price_to_vwap_ratio < 1.005)
        
        # Long: Weekly RSI oversold (<30) + price at VWAP + volume spike + above EMA50 (uptrend)
        if position == 0 and rsi_1w_i < 30 and near_vwap and volume_spike and close[i] > ema_50[i]:
            position = 1
            signals[i] = position_size
        # Short: Weekly RSI overbought (>70) + price at VWAP + volume spike + below EMA50 (downtrend)
        elif position == 0 and rsi_1w_i > 70 and near_vwap and volume_spike and close[i] < ema_50[i]:
            position = -1
            signals[i] = -position_size
        # Exit: RSI returns to neutral zone (40-60) or opposite extreme
        elif position != 0:
            if position == 1 and (rsi_1w_i > 40 or rsi_1w_i > 70):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (rsi_1w_i < 60 or rsi_1w_i < 30):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_WeeklyRSI_VWAP_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0