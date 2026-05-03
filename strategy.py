#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) extreme reversal with 4h Donchian(20) trend filter and volume confirmation
# In bull markets: buy when RSI(2) < 10 (oversold) + price > 4h Donchian upper + volume spike
# In bear markets: sell when RSI(2) > 90 (overbought) + price < 4h Donchian lower + volume spike
# RSI(2) captures short-term exhaustion; Donchian(20) provides structural trend bias
# Volume spike (>2.0x 24-period EMA) confirms institutional participation
# Session filter (08-20 UTC) reduces noise outside active trading hours
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "1h_RSI2_4hDonchian20_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate RSI(2) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation: 24-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 for RSI(2) calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ema_24[i]) or not in_session.iloc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 24-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_24[i])
        
        if position == 0:
            # Long: RSI(2) < 10 (extreme oversold) + price > 4h Donchian high + volume spike
            if (rsi[i] < 10 and 
                close[i] > donchian_high_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (extreme overbought) + price < 4h Donchian low + volume spike
            elif (rsi[i] > 90 and 
                  close[i] < donchian_low_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(2) > 50 (neutral) OR price < 4h Donchian low
            if rsi[i] > 50 or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(2) < 50 (neutral) OR price > 4h Donchian high
            if rsi[i] < 50 or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals