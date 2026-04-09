#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + volume confirmation + weekly ADX regime filter
# In weekly trending regime (ADX > 25): trade Donchian breakouts in trend direction
# In weekly ranging regime (ADX <= 25): fade Donchian breakouts (mean reversion)
# Uses 1d Donchian channels for structure, volume spike for confirmation, 1w ADX for regime
# Position size 0.25 to limit drawdown, discrete levels to minimize fee churn
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in bull/bear: adapts to weekly regime via ADX filter

name = "1d_1w_donchian_volume_adx_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ADX regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for regime detection
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr0 = high_1w[i] - low_1w[i]
        tr1 = abs(high_1w[i] - close_1w[i-1])
        tr2 = abs(low_1w[i] - close_1w[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1w), np.nan)
    minus_di_14 = np.full(len(df_1w), np.nan)
    dx_14 = np.full(len(df_1w), np.nan)
    
    for i in range(14, len(df_1w)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align weekly ADX to daily timeframe
    adx_14_1d = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < lookback - 1:
            continue
        highest_high[i] = np.max(high[max(0, i-lookback+1):i+1])
        lowest_low[i] = np.min(low[max(0, i-lookback+1):i+1])
    
    # Calculate volume spike (2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            continue
        vol_ma[i] = np.mean(volume[max(0, i-20):i+1])
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(adx_14_1d[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_1d[i]
        vol_spike = volume_spike[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Weekly trending regime
                # Exit when price closes below Donchian middle or volume drops
                middle = (highest_high[i] + lowest_low[i]) / 2
                if close[i] < middle:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Weekly ranging regime
                # Exit when price returns to Donchian lower band (mean reversion)
                if close[i] <= lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Weekly trending regime
                # Exit when price closes above Donchian middle or volume drops
                middle = (highest_high[i] + lowest_low[i]) / 2
                if close[i] > middle:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Weekly ranging regime
                # Exit when price returns to Donchian upper band (mean reversion)
                if close[i] >= highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime and volume confirmation
            if vol_spike:  # Only trade on volume spikes
                if adx > 25:  # Weekly trending regime - follow breakout
                    # Go long on breakout above upper band
                    if high[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.25
                    # Go short on breakdown below lower band
                    elif low[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.25
                else:  # Weekly ranging regime - fade breakout (mean reversion)
                    # Go long on breakdown below lower band (expect reversion up)
                    if low[i] < lowest_low[i]:
                        position = 1
                        signals[i] = 0.25
                    # Go short on breakout above upper band (expect reversion down)
                    elif high[i] > highest_high[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals