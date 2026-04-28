#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below
# with 1d EMA34 uptrend and volume > 2.0x 20-bar average. Short when %R crosses below -20 from above
# with 1d EMA34 downtrend and volume spike. Works in both bull/bear markets by fading extremes
# with trend filter. Target 12-37 trades/year via tight reversal conditions.
# Uses 12h primary timeframe to minimize fee drag while capturing multi-day swings.

name = "12h_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 12h timeframe (completed daily levels only)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R on 12h data (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 14)  # volume MA20, EMA34, Williams %R need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        ema34_val = ema34_1d_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below AND 1d EMA34 uptrend AND volume spike
            if wr > -80 and wr_prev <= -80 and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Williams %R crosses below -20 from above AND 1d EMA34 downtrend AND volume spike
            elif wr < -20 and wr_prev >= -20 and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on Williams %R crossing above -20 (overbought) or stoploss
            # ATR-based stoploss: 2.0 * ATR below entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on overbought condition or stoploss
            if wr > -20 or price < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on Williams %R crossing below -80 (oversold) or stoploss
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on oversold condition or stoploss
            if wr < -80 or price > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals