#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly RSI for mean reversion and daily volume for confirmation
# - Uses 1w RSI(14) to identify oversold/overbought conditions
# - Uses 1d volume spike (2x 20-day average) for confirmation
# - Enters long when 1w RSI < 30 and price closes above 1d open with volume spike
# - Enters short when 1w RSI > 70 and price closes below 1d open with volume spike
# - Exits when RSI returns to neutral zone (40-60) or opposite signal occurs
# - Designed to capture mean reversion in weekly extremes with daily confirmation
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wRSI_1dVolume_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w RSI (14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period):
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        if len(gain) < period:
            return np.full_like(gain, 50.0)
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = wilders_rsi(gain, loss, 14)
    
    # Align 1w RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily volume spike (2x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w RSI oversold (<30) and price closes above open with volume spike
            if rsi_1w_aligned[i] < 30 and close[i] > open_[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1w RSI overbought (>70) and price closes below open with volume spike
            elif rsi_1w_aligned[i] > 70 and close[i] < open_[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40) or opposite signal
            if rsi_1w_aligned[i] >= 40 or (rsi_1w_aligned[i] > 70 and close[i] < open_[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60) or opposite signal
            if rsi_1w_aligned[i] <= 60 or (rsi_1w_aligned[i] < 30 and close[i] > open_[i] and volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals