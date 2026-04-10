#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation
# - Donchian breakout: price breaks above/below 20-period high/low on 4h
# - Trend filter: 12h HMA(21) slope positive/negative for trend direction
# - Volume confirmation: current 4h volume > 1.8x 20-period average
# - Entry logic:
#   * Long: price > DonchianUpper(20) AND 12h HMA sloping up AND volume spike
#   * Short: price < DonchianLower(20) AND 12h HMA sloping down AND volume spike
# - Exit: ATR(14) trailing stop (2.5x) on 4h timeframe
# - Weekly regime filter: only trade in direction of weekly EMA50 trend
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 400 total
# - Donchian breakouts capture strong momentum moves, HMA filter avoids counter-trend trades,
#   volume confirmation ensures conviction, ATR stop manages risk

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.array([wma(close_12h[i-half_len+1:i+1], half_len) if i >= half_len-1 else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i-21+1:i+1], 21) if i >= 20 else np.nan 
                         for i in range(len(close_12h))])
    raw_hma = 2 * wma_half - wma_full
    hma_12h = np.array([wma(raw_hma[i-sqrt_len+1:i+1], sqrt_len) if i >= sqrt_len-1 else np.nan 
                        for i in range(len(raw_hma))])
    
    # HMA slope: positive if current > previous
    hma_slope = np.diff(hma_12h, prepend=np.nan)
    hma_slope_up = hma_slope > 0
    hma_slope_down = hma_slope < 0
    
    # Pre-compute weekly EMA(50) for regime filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 4h timeframe
    hma_slope_up_aligned = align_htf_to_ltf(prices, df_12h, hma_slope_up)
    hma_slope_down_aligned = align_htf_to_ltf(prices, df_12h, hma_slope_down)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume and its 20-period moving average
    volume_4h = prices['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_slope_up_aligned[i]) or np.isnan(hma_slope_down_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(volume_ma_20_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h volume for filter
        volume_4h_current = volume_4h[i]
        
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        volume_spike = volume_4h_current > 1.8 * volume_ma_20_4h[i]
        
        # Weekly regime filter
        weekly_uptrend = close_4h[i] > ema_50_aligned[i]
        weekly_downtrend = close_4h[i] < ema_50_aligned[i]
        
        close_price = close_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + HMA up + volume spike + weekly uptrend
            if (close_price > donchian_upper[i] and 
                hma_slope_up_aligned[i] and 
                volume_spike and 
                weekly_uptrend):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + HMA down + volume spike + weekly downtrend
            elif (close_price < donchian_lower[i] and 
                  hma_slope_down_aligned[i] and 
                  volume_spike and 
                  weekly_downtrend):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals