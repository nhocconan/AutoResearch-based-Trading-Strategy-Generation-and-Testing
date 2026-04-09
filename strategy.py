#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla pivot levels with volume confirmation and ATR trailing stop
# - Uses 1w HTF for Camarilla pivot levels (based on completed weekly candles)
# - Long when price touches Camarilla L3 level with volume > 1.8x 20-period average and closes above open
# - Short when price touches Camarilla H3 level with volume > 1.8x 20-period average and closes below open
# - ATR(14) trailing stop: exit long at 2.5x ATR below highest high since entry, exit short at 2.5x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels adapt to volatility, volume and price action confirmation filters false signals
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1w_camarilla_volume_priceaction_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on previous week)
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    hl_range = high_1w - low_1w
    camarilla_h3 = close_1w + 1.0 * hl_range
    camarilla_l3 = close_1w - 1.0 * hl_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume and price action confirmation
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        bullish_candle = close[i] > open_price[i]  # Close above open
        bearish_candle = close[i] < open_price[i]  # Close below open
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla L3/H3 touch with volume and price action confirmation
            if volume_confirmed:
                # Long entry: price touches or goes below L3 with bullish close
                if low[i] <= camarilla_l3_aligned[i] and bullish_candle:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price touches or goes above H3 with bearish close
                elif high[i] >= camarilla_h3_aligned[i] and bearish_candle:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals