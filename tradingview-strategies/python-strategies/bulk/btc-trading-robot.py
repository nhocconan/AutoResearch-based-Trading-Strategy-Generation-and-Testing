#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTC Trading Robot"
timeframe = "1m"
leverage = 1

def generate_signals(prices):
    """
    Generates trading signals based on local breakout logic with trailing stops.
    Returns a numpy array of length len(prices) with values: 1 (Long), -1 (Short), 0 (Flat).
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Constants adapted from Pine Script inputs
    # TPasPctBTC 0.2 -> 0.002 (0.2%)
    # SLasPctBTC 0.1 -> 0.001 (0.1%) - Calculated but not used in exit logic provided
    # TSLTgrasPctofTPBTC 7.0 -> 0.07 (7% of TP)
    TP_PCT = 0.002
    TRAIL_PCT_OF_TP = 0.07
    BARS_N = 5
    LOOKBACK = BARS_N * 2 + 1
    
    # Time Filter Inputs (0 = disabled)
    SH_INPUT = 0
    EH_INPUT = 0
    
    # Precompute rolling highs/lows
    # min_periods=1 mimics ta.highest behavior on initial bars
    local_high = prices['high'].rolling(window=LOOKBACK, min_periods=1).max().values
    local_low = prices['low'].rolling(window=LOOKBACK, min_periods=1).min().values
    
    # Repo data already stores UTC-compatible datetimes; do not force ms coercion.
    hours = pd.to_datetime(prices['open_time'], utc=True).dt.hour.values
    
    # State variables
    position = 0  # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    
    for i in range(n):
        # Assign current signal based on state entering this bar
        signals[i] = position
        
        close = prices['close'].iloc[i]
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        
        # Time Filter Check
        in_session = True
        if SH_INPUT != 0 and hours[i] < SH_INPUT:
            in_session = False
        if EH_INPUT != 0 and hours[i] >= EH_INPUT:
            in_session = False
            
        # Calculate dynamic parameters based on current close
        tp_dist = close * TP_PCT
        trail_dist = tp_dist * TRAIL_PCT_OF_TP
        trail_trigger = tp_dist * TRAIL_PCT_OF_TP
        order_buffer = tp_dist / 2.0
        
        # Exit Logic (Check before updating position for next bar)
        next_position = position
        
        if position == 1:  # Long
            # Update max price
            if close > max_price_since_entry:
                max_price_since_entry = close
            
            # Check Trailing Stop
            # Stop level = max_price - trail_dist
            stop_level = max_price_since_entry - trail_dist
            
            # Check Dynamic TP
            # Limit level = entry_price + tp_dist
            tp_level = entry_price + tp_dist
            
            # Check Profit Trigger for Trailing Stop
            profit = close - entry_price
            
            # Exit if low hits stop or high hits TP
            # Approximation: If hit intrabar, close position for next bar
            if low <= stop_level or high >= tp_level:
                next_position = 0
            # Optional: Activate trailing only after trigger profit met
            # Pine logic applies exit order always once condition met, 
            # but stop price moves. Here we simplify to immediate exit if levels hit.
            
        elif position == -1:  # Short
            # Update min price
            if close < min_price_since_entry:
                min_price_since_entry = close
                
            # Check Trailing Stop
            # Stop level = min_price + trail_dist
            stop_level = min_price_since_entry + trail_dist
            
            # Check Dynamic TP
            # Limit level = entry_price - tp_dist
            tp_level = entry_price - tp_dist
            
            # Exit if high hits stop or low hits TP
            if high >= stop_level or low <= tp_level:
                next_position = 0
                
        # Entry Logic (Only if flat)
        if position == 0 and next_position == 0 and in_session:
            lh = local_high[i]
            ll = local_low[i]
            
            # Long Entry: close < localHigh - buffer
            # Order would be stop at localHigh. 
            # We enter next bar if condition met this bar.
            if close < (lh - order_buffer):
                next_position = 1
                entry_price = close
                max_price_since_entry = close
                
            # Short Entry: close > localLow + buffer
            elif close > (ll + order_buffer):
                next_position = -1
                entry_price = close
                min_price_since_entry = close
        
        # Update state for next bar
        position = next_position
        
    return signals
