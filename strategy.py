#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 level AND 1w close > 1w EMA20 (uptrend)
# - Short when price breaks below Camarilla L4 level AND 1w close < 1w EMA20 (downtrend)
# - Volume confirmation: 12h volume > 1.5x 20-period 12h volume SMA
# - Exit: price touches Camarilla L3/H3 levels or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Uses 1w EMA for trend filter to avoid counter-trend trades in strong trends
# - Camarilla levels provide precise intraday support/resistance from prior 1d range

name = "12h_1w_camarilla_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load HTF data: 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return signals
    
    # Calculate 1w EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Track entry price for stoploss (optional)
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after volume SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Need prior 1d high/low for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from prior 1d bar
        # Camarilla uses prior day's range
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_close = close[i-1]
        
        if np.isnan(prior_high) or np.isnan(prior_low) or np.isnan(prior_close):
            signals[i] = 0.0
            continue
            
        range_val = prior_high - prior_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        # H4 = close + 1.1 * range * 1.1 / 2
        # L4 = close - 1.1 * range * 1.1 / 2
        # H3 = close + 1.1 * range * 1.1 / 4
        # L3 = close - 1.1 * range * 1.1 / 4
        camarilla_h4 = prior_close + 1.1 * range_val * 1.1 / 2
        camarilla_l4 = prior_close - 1.1 * range_val * 1.1 / 2
        camarilla_h3 = prior_close + 1.1 * range_val * 1.1 / 4
        camarilla_l3 = prior_close - 1.1 * range_val * 1.1 / 4
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter: 1w close vs EMA20
        trend_up = close_1w[-1] > ema_20_1w[-1] if len(close_1w) > 0 else False  # Simplified - use aligned
        trend_down = close_1w[-1] < ema_20_1w[-1] if len(close_1w) > 0 else False
        
        # Use aligned EMA for proper timing
        trend_up = close[i] > ema_20_1w_aligned[i]
        trend_down = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:  # Flat - look for entry
            # Long: break above H4 with uptrend and volume
            if close[i] > camarilla_h4 and trend_up and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short: break below L4 with downtrend and volume
            elif close[i] < camarilla_l4 and trend_down and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit: touch L3 or opposite break below L4 with volume
            exit_condition = (close[i] <= camarilla_l3) or \
                           (close[i] < camarilla_l4 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit: touch H3 or opposite break above H4 with volume
            exit_condition = (close[i] >= camarilla_h3) or \
                           (close[i] > camarilla_h4 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals