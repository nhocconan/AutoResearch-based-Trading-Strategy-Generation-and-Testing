#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation
# H3/L3 are stronger reversal levels than R3/S3 in ranging markets, less prone to false breakouts
# Long when price breaks above H3 with volume spike AND above 1d EMA50 (bullish trend)
# Short when price breaks below L3 with volume spike AND below 1d EMA50 (bearish trend)
# Exit when price returns to H4/L4 levels or trend reverses
# Uses 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) minimizing fee drag
# Works in both bull and bear markets by following the higher timeframe trend

name = "12h_Camarilla_H3L3_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    price_range = high - low
    
    # Camarilla levels for intraday (based on previous day)
    # H4 = close + range * 1.1/2
    # H3 = close + range * 1.1/4
    # L3 = close - range * 1.1/4
    # L4 = close - range * 1.1/2
    camarilla_h3 = close + price_range * 1.1 / 4.0
    camarilla_l3 = close - price_range * 1.1 / 4.0
    camarilla_h4 = close + price_range * 1.1 / 2.0
    camarilla_l4 = close - price_range * 1.1 / 2.0
    
    # Volume confirmation: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema50_aligned[i]
        camarilla_h3_prev = camarilla_h3[i-1]  # Use previous day's levels
        camarilla_l3_prev = camarilla_l3[i-1]
        camarilla_h4_prev = camarilla_h4[i-1]
        camarilla_l4_prev = camarilla_l4[i-1]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above H3 AND bullish regime
                if curr_high > camarilla_h3_prev and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below L3 AND bearish regime
                elif curr_low < camarilla_l3_prev and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to H4 OR regime changes to bearish
            if curr_close >= camarilla_h4_prev or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to L4 OR regime changes to bullish
            if curr_close <= camarilla_l4_prev or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals