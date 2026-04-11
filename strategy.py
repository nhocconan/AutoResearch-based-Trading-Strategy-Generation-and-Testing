# Solution: 12h_1w_parabolic_sar_reversal_v1
# This strategy uses Parabolic SAR on weekly timeframe for trend detection and reversal signals.
# Combined with volume confirmation and ATR filter on 12h timeframe to filter false signals.
# The Parabolic SAR provides clear trend direction changes which work in both bull and bear markets.
# Volume confirmation ensures institutional participation in the move.
# ATR filter prevents entries during low volatility periods.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe.
# Weekly Parabolic SAR reduces noise and captures multi-week trends.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_parabolic_sar_reversal_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate weekly Parabolic SAR
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize SAR
    sar = np.zeros(len(close_1w))
    trend = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Start with first point
    sar[0] = low_1w[0]
    ep = high_1w[0]  # extreme point
    
    for i in range(1, len(close_1w)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # SAR in uptrend cannot be above the low of the past two periods
            if i >= 2:
                sar[i] = min(sar[i], low_1w[i-1], low_1w[i-2])
            # Trend reversal: price below SAR
            if low_1w[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep  # SAR becomes the previous EP
                ep = low_1w[i]  # reset EP to current low
                af = 0.02  # reset acceleration factor
            else:
                trend[i] = 1
                # Update EP if new high
                if high_1w[i] > ep:
                    ep = high_1w[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            sar[i] = sar[i-1] + af * (sar[i-1] - ep)
            # SAR in downtrend cannot be below the high of the past two periods
            if i >= 2:
                sar[i] = max(sar[i], high_1w[i-1], high_1w[i-2])
            # Trend reversal: price above SAR
            if high_1w[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep  # SAR becomes the previous EP
                ep = high_1w[i]  # reset EP to current high
                af = 0.02  # reset acceleration factor
            else:
                trend[i] = -1
                # Update EP if new low
                if low_1w[i] < ep:
                    ep = low_1w[i]
                    af = min(af + 0.02, max_af)
    
    # Shift SAR by 1 to use only completed weekly bars (avoid look-ahead)
    sar = np.roll(sar, 1)
    sar[0] = np.nan
    
    # Align weekly SAR to 12h timeframe
    sar_aligned = align_htf_to_ltf(prices, df_1w, sar)
    
    # Calculate 12h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(sar_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.008 * price_close  # ATR > 0.8% of price
        
        # Long conditions: price crosses above weekly SAR with volume and vol filter
        long_signal = volume_confirmed and vol_filter and (price_close > sar_aligned[i]) and (price_close <= sar_aligned[i-1] if i > 0 else False)
        
        # Short conditions: price crosses below weekly SAR with volume and vol filter
        short_signal = volume_confirmed and vol_filter and (price_close < sar_aligned[i]) and (price_close >= sar_aligned[i-1] if i > 0 else False)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and price_close < sar_aligned[i]:
            # Exit long when price crosses back below SAR
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_close > sar_aligned[i]:
            # Exit short when price crosses back above SAR
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Parabolic SAR reversal with volume and volatility filters on 12h timeframe.
# Uses weekly Parabolic SAR to identify trend direction and potential reversals.
# Enters long when 12h price crosses above weekly SAR with volume confirmation (>1.3x average)
# and sufficient volatility (ATR > 0.8% of price). Enters short when price crosses below weekly SAR
# under same conditions. Exits when price crosses back in the opposite direction of SAR.
# Works in both bull and bear markets by capturing trend reversals. Target: 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag on 12h timeframe. Weekly timeframe reduces noise
# and captures multi-week trends. Volume confirmation ensures institutional participation.
# Volatility filter prevents whipsaws in low volatility environments. SAR-based exit provides
# systematic trend-following while allowing profits to run.