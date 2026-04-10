#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# - Primary: 6h Elder Ray (EMA13-based Bull Power = Close - EMA13, Bear Power = EMA13 - Close)
# - HTF: 1w EMA50 for major trend direction + 1w volume confirmation (current volume > 1.5x 20-period MA)
# - Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + price > 1w EMA50 + volume confirmation
# - Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) + price < 1w EMA50 + volume confirmation
# - Exit: Elder Ray divergence (Bull Power < 0 for long, Bear Power < 0 for short) OR price crosses EMA13
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: 1w EMA50 filters major trend, Elder Ray captures momentum shifts, volume confirms conviction
# - Target: 75-150 trades over 4 years (19-37/year) to stay within fee drag limits

name = "6h_1w_elderray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for EMA and volume
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components (6h)
    bull_power_6h = close_6h - ema13_6h  # Close - EMA13
    bear_power_6h = ema13_6h - close_6h  # EMA13 - Close
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume moving average (20-period) for volume confirmation
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Also align 1w volume for direct use
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ma_20_1w_aligned[i]) or
            np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period MA
        volume_confirm = volume_1w_aligned[i] > 1.5 * volume_ma_20_1w_aligned[i]
        
        # Elder Ray signals
        bullish_momentum = bull_power_6h[i] > 0 and bear_power_6h[i] < 0  # Bull Power > 0 AND Bear Power < 0
        bearish_momentum = bull_power_6h[i] < 0 and bear_power_6h[i] > 0  # Bull Power < 0 AND Bear Power > 0
        
        # Trend filter: price relative to 1w EMA50
        above_trend = close_6h[i] > ema50_1w_aligned[i]
        below_trend = close_6h[i] < ema50_1w_aligned[i]
        
        # Exit conditions: Elder Ray divergence or price crosses EMA13
        exit_long = bull_power_6h[i] <= 0 or close_6h[i] <= ema13_6h[i]
        exit_short = bear_power_6h[i] <= 0 or close_6h[i] >= ema13_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish momentum + above 1w EMA50 + volume confirmation
            if bullish_momentum and above_trend and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish momentum + below 1w EMA50 + volume confirmation
            elif bearish_momentum and below_trend and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Elder Ray divergence or price crosses EMA13
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals