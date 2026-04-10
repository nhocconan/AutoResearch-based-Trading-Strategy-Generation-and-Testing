#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter
# - Bollinger Band Width(20,2) on 6h: identifies low volatility squeeze
# - BBW percentile < 20% = squeeze condition (low volatility)
# - Breakout on expansion: price closes outside BB(20,2) AND BBW rising
# - 1d ADX(14) > 25 = strong trend filter (avoid false breakouts in range)
# - Long: 6h BB squeeze + price closes above upper BB AND 1d ADX > 25
# - Short: 6h BB squeeze + price closes below lower BB AND 1d ADX > 25
# - Exit: price returns to middle BB (20-period SMA) OR BBW contracts back below 30th percentile
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years) to avoid fee drag
# - Works in both bull/bear: ADX filter ensures we only trade strong trends,
#   BB squeeze captures volatility expansion after consolidation

name = "6h_1d_bb_squeeze_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]),
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where(np.isnan(dx_1d), 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h Bollinger Bands (20,2)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Middle band = 20-period SMA
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    # Bollinger Band Width
    bb_width = (upper_bb - lower_bb) / sma_20
    # BB Width percentile rank (lookback 50 periods for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    bb_width_percentile = np.where(np.isnan(bb_width_percentile), 50, bb_width_percentile)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup for BBW percentile
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(adx_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Squeeze condition: BBW percentile < 20 (low volatility)
        squeeze_condition = bb_width_percentile[i] < 20
        
        # Breakout conditions
        breakout_up = close[i] > upper_bb[i]  # Close above upper BB
        breakout_down = close[i] < lower_bb[i]  # Close below lower BB
        
        # BBW expansion: current BBW > previous BBW (volatility increasing)
        bbw_expanding = bb_width[i] > bb_width[i-1] if i > 0 else False
        
        # 1d ADX trend filter: ADX > 25 = strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions
        long_entry = squeeze_condition and breakout_up and bbw_expanding and strong_trend
        short_entry = squeeze_condition and breakout_down and bbw_expanding and strong_trend
        
        # Exit conditions
        # Exit long: price returns to middle BB OR BBW contracts back below 30th percentile
        exit_long = (position == 1 and 
                    (close[i] <= sma_20[i] or bb_width_percentile[i] < 30))
        # Exit short: price returns to middle BB OR BBW contracts back below 30th percentile
        exit_short = (position == -1 and 
                     (close[i] >= sma_20[i] or bb_width_percentile[i] < 30))
        
        if position == 0:  # Flat - look for new entries
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals