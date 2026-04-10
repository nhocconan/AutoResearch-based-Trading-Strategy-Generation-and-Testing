#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Primary: 6h Camarilla pivot levels calculated from prior 1d OHLC
# - HTF: 1d EMA(50) for trend direction (only trade with trend)
# - HTF: 1d volume MA(20) for volume confirmation
# - Long: price breaks above Camarilla R4 + price > 1d EMA50 + 1d volume > 1.2x 20-period MA
# - Short: price breaks below Camarilla S4 + price < 1d EMA50 + 1d volume > 1.2x 20-period MA
# - Exit: price crosses Camarilla pivot point (PP) (mean reversion exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: trend filter avoids counter-trend, volume confirmation ensures validity
# - Target: 75-200 trades over 4 years (19-50/year) to stay within fee drag limits

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Camarilla pivot levels from prior 1d OHLC
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = np.concatenate([[np.nan], high_1d[:-1]])  # Previous day's high
    prior_low = np.concatenate([[np.nan], low_1d[:-1]])    # Previous day's low
    prior_close = np.concatenate([[np.nan], close_1d[:-1]]) # Previous day's close
    
    # Calculate pivot point (PP) and ranges
    pp = (prior_high + prior_low + prior_close) / 3
    range_hl = prior_high - prior_low
    
    # Camarilla levels
    r4 = pp + range_hl * 1.1 / 2
    r3 = pp + range_hl * 1.1 / 4
    s3 = pp - range_hl * 1.1 / 4
    s4 = pp - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 6h)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Regime conditions
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close_6h[i] > ema_50_1d_aligned[i]
        price_below_ema = close_6h[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 1.2x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 1.2 * volume_ma_20_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = close_6h[i] > r4_aligned[i]
        breakout_down = close_6h[i] < s4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Camarilla breakout up + uptrend + volume confirmation
            if (breakout_up and price_above_ema and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Camarilla breakout down + downtrend + volume confirmation
            elif (breakout_down and price_below_ema and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price crosses Camarilla pivot point (PP) (mean reversion exit)
            if position == 1:  # Long position
                exit_condition = close_6h[i] < pp_aligned[i]  # Crossed below PP
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = close_6h[i] > pp_aligned[i]  # Crossed above PP
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals