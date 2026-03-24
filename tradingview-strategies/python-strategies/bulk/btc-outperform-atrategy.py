#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC outperform atrategy"
timeframe = "1w"
leverage = 1

def generate_signals(prices):
    """
    Generates trading signals based on weekly vs 3-month close comparison.
    Uses shifted 3-month resampling logic to avoid lookahead bias.
    """
    df = prices.copy()
    
    # Repo data already stores UTC-compatible datetimes; do not force ms coercion.
    df['dt'] = pd.to_datetime(df['open_time'], utc=True)
    df.set_index('dt', inplace=True)
    
    # Weekly price is the current close (since input timeframe is 1w)
    weekly_price = df['close']
    
    # Monthly (3M) price logic:
    # 1. Resample to 3-month frequency to get quarter-end closes.
    # 2. Shift by 1 to ensure we use the previous completed quarter's close.
    # 3. Reindex and forward-fill to align with weekly bars.
    # Fix: Use '3ME' instead of '3M' for pandas 2.2+ compatibility.
    quarterly_close = df['close'].resample('3ME').last()
    quarterly_close_shifted = quarterly_close.shift(1)
    monthly_price = quarterly_close_shifted.reindex(df.index).ffill()
    
    # Calculate conditions
    # Comparisons with NaN result in False, defaulting to 0 signal
    long_condition = weekly_price > monthly_price
    short_condition = monthly_price > weekly_price
    
    # Initialize signals array
    signals = np.zeros(len(df), dtype=int)
    
    # Assign signals
    # Convert boolean series to numpy arrays for indexing
    signals[long_condition.values] = 1
    signals[short_condition.values] = -1
    
    return signals
