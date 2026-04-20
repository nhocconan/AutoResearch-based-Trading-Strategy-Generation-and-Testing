#1d_1w_WeeklyTrend_DailyBreakout_v3
# Hypothesis: Weekly trend (1w EMA50) + Daily breakout (Donchian20) + Volume confirmation
# Works in bull (trend + breakout) and bear (avoid counter-trend via weekly filter)
# Target: 20-50 trades/year, low frequency to minimize fee drag
# Timeframe: 1d, HTF: 1w

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyTrend_DailyBreakout_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly: Trend filter (EMA50) ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily: Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        ema_val = ema50_1w_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(donch_high_val) or np.isnan(donch_low_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price breaks above Donchian high + volume confirmation
            if (close_val > ema_val and           # Weekly uptrend (price above weekly EMA50)
                close_val > donch_high_val and    # Break above Donchian high (20)
                vol_ratio_val > 1.5):             # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price breaks below Donchian low + volume confirmation
            elif (close_val < ema_val and         # Weekly downtrend (price below weekly EMA50)
                  close_val < donch_low_val and   # Break below Donchian low (20)
                  vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or breakdown below Donchian low
            if (close_val < ema_val or           # Weekly trend turned down
                close_val < donch_low_val):      # Broke below Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or breakout above Donchian high
            if (close_val > ema_val or           # Weekly trend turned up
                close_val > donch_high_val):     # Broke above Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals