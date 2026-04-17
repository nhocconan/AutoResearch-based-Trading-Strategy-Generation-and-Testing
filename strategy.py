#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX(14) + ATR(14) breakout with weekly EMA200 trend filter and volume confirmation.
# Uses weekly EMA200 for long-term trend direction, daily ADX for trend strength,
# and ATR-based breakout from the previous day's range.
# Volume spike confirms breakout strength.
# Designed to capture strong trending moves with low turnover in both bull and bear markets.
# Target: 15-25 trades/year to stay within optimal range for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily
    ema200_1d = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily ATR(14) for volatility and breakout calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ADX(14) for trend strength
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 200-period EMA (weekly) + 14-period ADX/ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d[i]) or 
            np.isnan(adx14[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA200
        price_above_ema200 = close[i] > ema200_1d[i]
        price_below_ema200 = close[i] < ema200_1d[i]
        
        # Trend strength filter: ADX > 25
        strong_trend = adx14[i] > 25
        
        # Volatility filter: ATR > 0
        volatility_ok = atr14[i] > 0
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # ATR-based breakout levels from previous day
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_range = prev_high - prev_low
            
            # Long breakout: above previous high + 0.5 * ATR
            long_breakout = close[i] > (prev_high + 0.5 * atr14[i])
            
            # Short breakout: below previous low - 0.5 * ATR
            short_breakout = close[i] < (prev_low - 0.5 * atr14[i])
        else:
            long_breakout = False
            short_breakout = False
        
        if position == 0:
            # Long: Breakout above previous day's high with volume, trend strength, and above weekly EMA200
            if (long_breakout and price_above_ema200 and strong_trend and volatility_ok and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Breakout below previous day's low with volume, trend strength, and below weekly EMA200
            elif (short_breakout and price_below_ema200 and strong_trend and volatility_ok and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close below previous day's low OR ADX falls below 20 (trend weakening)
            if (close[i] < low[i-1]) or (adx14[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above previous day's high OR ADX falls below 20 (trend weakening)
            if (close[i] > high[i-1]) or (adx14[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_ADX14_ATR14_Breakout_EMA200_Volume"
timeframe = "1d"
leverage = 1.0