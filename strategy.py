#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Camarilla pivot levels with weekly EMA trend filter and volume confirmation
# - Long when price touches Camarilla L3 support AND weekly EMA(21) is rising AND volume > 1.5x 20-day volume SMA
# - Short when price touches Camarilla H3 resistance AND weekly EMA(21) is falling AND volume > 1.5x 20-day volume SMA
# - Exit: Price reaches Camarilla H4 (for longs) or L4 (for shorts) or opposite pivot touch with volume
# - Position sizing: 0.25 discrete level
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Weekly EMA filter ensures trades align with higher timeframe trend, reducing counter-trend whipsaw
# - Camarilla pivots provide mathematically derived support/resistance levels that work across market regimes

name = "1d_1w_camarilla_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Daily Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Calculate 20-day volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load Weekly EMA(21) data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:  # Need enough data for EMA(21)
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate EMA slope (rising/falling) - using 3-period change
    ema_slope = np.diff(ema_21_1w_aligned, prepend=np.nan)
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Track entry price for stoploss logic
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after volume SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: daily volume > 1.5x 20-day volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Pivot touch conditions (with small tolerance for wicks)
        tolerance = 0.001  # 0.1% tolerance for touch
        touch_h3 = abs(high[i] - camarilla_h3[i]) / camarilla_h3[i] <= tolerance
        touch_l3 = abs(low[i] - camarilla_l3[i]) / camarilla_l3[i] <= tolerance
        
        if position == 0:  # Flat - look for entry
            # Long: touch L3 support + weekly EMA rising + volume confirmation
            if touch_l3 and ema_rising[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short: touch H3 resistance + weekly EMA falling + volume confirmation
            elif touch_h3 and ema_falling[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit: price reaches H4 target OR touches H3 with volume (failure)
            exit_condition = (high[i] >= camarilla_h4[i]) or \
                           (touch_h3 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit: price reaches L4 target OR touches L3 with volume (failure)
            exit_condition = (low[i] <= camarilla_l4[i]) or \
                           (touch_l3 and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals