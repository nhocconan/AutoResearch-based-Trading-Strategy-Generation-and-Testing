#!/usr/bin/env python3
# 1h_ema_rsi_pullback_4h1d_v1
# Hypothesis: 1h strategy using 4h EMA21 trend direction and daily RSI(14) for mean reversion entries.
# Long: 4h EMA21 uptrend, price pulls back to 1h EMA50 with RSI(14) < 30, volume > 1.2x 20-period average.
# Short: 4h EMA21 downtrend, price pulls back to 1h EMA50 with RSI(14) > 70, volume > 1.2x 20-period average.
# Exit: Opposite EMA50 cross or RSI reaches extreme (RSI>70 for long, RSI<30 for short).
# Uses 4h EMA for trend filter (aligns with experiment's HTF preference) and daily RSI for timing.
# Session filter: 08-20 UTC to reduce noise trades.
# Position size: 0.20 (discrete level to minimize fee churn).
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_rsi_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h EMA50 for pullback target
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1h RSI(14) for entry timing
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when undefined
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for EMA21 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    close_4h = pd.Series(df_4h['close'].values)
    ema21_4h = close_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for RSI(14) momentum filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # 1d RSI(14) for momentum filter
    close_1d = pd.Series(df_1d['close'].values)
    delta_1d = close_1d.diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0.0)
    loss_1d = (-delta_1d).where(delta_1d < 0, 0.0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.fillna(50).values  # Neutral RSI when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (not in_session[i] or
            np.isnan(ema50[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]) or
            np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below EMA50 OR RSI reaches overbought (70)
            if close[i] < ema50[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price crosses above EMA50 OR RSI reaches oversold (30)
            if close[i] > ema50[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: 4h EMA21 uptrend, 1d RSI < 50 (bullish momentum), price at EMA50, RSI < 30, volume confirmed
            if (ema21_4h_aligned[i] > ema21_4h_aligned[i-1] and  # 4h EMA21 rising
                rsi_1d_aligned[i] < 50 and                      # 1d RSI not overbought
                abs(close[i] - ema50[i]) / ema50[i] < 0.005 and  # Price near EMA50 (within 0.5%)
                rsi[i] < 30 and                                 # 1h RSI oversold
                volume_confirmed):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h EMA21 downtrend, 1d RSI > 50 (bearish momentum), price at EMA50, RSI > 70, volume confirmed
            elif (ema21_4h_aligned[i] < ema21_4h_aligned[i-1] and  # 4h EMA21 falling
                  rsi_1d_aligned[i] > 50 and                      # 1d RSI not oversold
                  abs(close[i] - ema50[i]) / ema50[i] < 0.005 and  # Price near EMA50 (within 0.5%)
                  rsi[i] > 70 and                                 # 1h RSI overbought
                  volume_confirmed):
                position = -1
                signals[i] = -0.20
    
    return signals