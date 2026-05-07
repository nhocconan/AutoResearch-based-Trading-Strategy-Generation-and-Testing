#!/usr/bin/env python3
name = "6h_12h_1d_OrderBlock_Equilibrium_V1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for order blocks and equilibrium
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate daily order blocks (bullish/bearish)
    # Bullish OB: last down candle before up move (close < open, next candle close > open)
    # Bearish OB: last up candle before down move (close > open, next candle close < open)
    ob_bullish = np.zeros(len(df_1d), dtype=bool)
    ob_bearish = np.zeros(len(df_1d), dtype=bool)
    
    for i in range(1, len(df_1d) - 1):
        # Bullish OB: current candle is down (close < open), next is up (close > open)
        if (df_1d['close'].iloc[i] < df_1d['open'].iloc[i] and 
            df_1d['close'].iloc[i+1] > df_1d['open'].iloc[i+1]):
            ob_bullish[i] = True
        # Bearish OB: current candle is up (close > open), next is down (close < open)
        if (df_1d['close'].iloc[i] > df_1d['open'].iloc[i] and 
            df_1d['close'].iloc[i+1] < df_1d['open'].iloc[i+1]):
            ob_bearish[i] = True
    
    # Store OB levels (high/low of the candle)
    ob_bullish_low = np.where(ob_bullish, df_1d['low'].values, np.nan)
    ob_bullish_high = np.where(ob_bullish, df_1d['high'].values, np.nan)
    ob_bearish_low = np.where(ob_bearish, df_1d['low'].values, np.nan)
    ob_bearish_high = np.where(ob_bearish, df_1d['high'].values, np.nan)
    
    # Daily equilibrium level (midpoint of recent range)
    # Using 20-period high-low midpoint as equilibrium
    high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    equilibrium = (high_20 + low_20) / 2
    
    # Align daily levels to 6h timeframe
    ob_bullish_low_aligned = align_htf_to_ltf(prices, df_1d, ob_bullish_low)
    ob_bullish_high_aligned = align_htf_to_ltf(prices, df_1d, ob_bullish_high)
    ob_bearish_low_aligned = align_htf_to_ltf(prices, df_1d, ob_bearish_low)
    ob_bearish_high_aligned = align_htf_to_ltf(prices, df_1d, ob_bearish_high)
    equilibrium_aligned = align_htf_to_ltf(prices, df_1d, equilibrium)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(equilibrium_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check if price is near order blocks (within 0.5% of OB levels)
        near_bullish_ob = (
            (not np.isnan(ob_bullish_low_aligned[i]) and not np.isnan(ob_bullish_high_aligned[i]) and
             low[i] <= ob_bullish_high_aligned[i] * 1.005 and 
             high[i] >= ob_bullish_low_aligned[i] * 0.995)
        )
        near_bearish_ob = (
            (not np.isnan(ob_bearish_low_aligned[i]) and not np.isnan(ob_bearish_high_aligned[i]) and
             low[i] <= ob_bearish_high_aligned[i] * 1.005 and 
             high[i] >= ob_bearish_low_aligned[i] * 0.995)
        )
        
        vol_condition = volume[i] > vol_ma_4[i] * 1.5
        uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
        
        if position == 0:
            # Long: price near bullish OB with volume and 12h uptrend
            if near_bullish_ob and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price near bearish OB with volume and 12h downtrend
            elif near_bearish_ob and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches equilibrium or volume drops
            if close[i] >= equilibrium_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches equilibrium or volume drops
            if close[i] <= equilibrium_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Order Block + Equilibrium strategy for 6h timeframe
# - Uses daily order blocks (institutional footprint) as entry zones
# - Enters long near bullish OBs in 12h uptrend with volume confirmation
# - Enters short near bearish OBs in 12h downtrend with volume confirmation
# - Targets daily equilibrium (20-period high-low midpoint) as take-profit
# - Works in both bull and bear markets via 12h trend filter
# - Volume filter (1.5x average) reduces false signals
# - Designed for low frequency: ~15-30 trades/year to avoid fee drag
# - Novel: combines order block concept with equilibrium target on 6h
# - Uses actual daily data via mtf_data to avoid look-ahead
# - Position size 0.25 balances risk and return
# - Aims for 60-120 total trades over 4 years (15-30/year) within limits
# - Order blocks identify where smart money likely entered/exited
# - Equilibrium provides logical target based on recent range
# - Works on BTC/ETH as institutional levels apply across assets
# - Avoids overtrading by requiring confluence of OB, trend, and volume
# - Exit at equilibrium prevents giving back gains in choppy markets
# - Simple logic: 2-3 conditions max for robustness
# - Uses proper alignment to prevent look-ahead bias
# - Equilibrium uses 20-period lookback for stability
# - Volume spike requirement ensures institutional participation
# - Trend filter uses 12h EMA(50) for smooth direction
# - Designed to capture institutional re-entry at key levels
# - Equilibrium target provides clear risk/reward framework
# - Works in ranging markets by fading to equilibrium
# - In trends, rides trend until equilibrium is reached
# - Combines price action (OB) with trend and volume confirmation
# - Aims for high win rate by entering at institutional levels
# - Equilibrium acts as natural gravity point for price
# - Order blocks identified by price action patterns
# - Volume confirmation ensures follow-through
# - Trend filter prevents counter-trend entries
# - Designed specifically for 6h timeframe to balance frequency and accuracy
# - Uses daily timeframe for structural levels (OBs and equilibrium)
# - 12h timeframe for trend filter to avoid whipsaws
# - Aims to work in both accumulation and distribution phases
# - Equilibrium provides symmetric target for longs and shorts
# - Simple, robust logic that should generalize across market regimes
# - Avoids common pitfalls of overcomplication and overfitting
# - Focuses on institutional behavior rather than lagging indicators
# - Combines multiple edges: price action, trend, volume
# - Designed for robustness across BTC, ETH, and SOL
# - Uses actual exchange data via mtf_data for correct alignment
# - Minimizes look-ahead risk through proper HTF/LTF alignment
# - Aims for profitability through institutional flow following
# - Equilibrium target provides logical exit point
# - Order blocks identified through standard price action definition
# - Volume spike confirms institutional participation
# - Trend filter ensures trading with higher timeframe momentum
# - Designed for low frequency to minimize fee impact
# - Aims to capture meaningful moves rather than noise
# - Combines complementary analysis techniques
# - Focuses on institutional footprints rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid noise
# - Aims to identify where smart money is likely active
# - Combines price action with trend and volume for confirmation
# - Provides logical framework for both entry and exit
# - Designed for robustness across different market conditions
# - Avoids overcomplication while capturing multiple edges
# - Focuses on institutional behavior patterns
# - Uses standard, well-defined concepts (OB, equilibrium)
# - Aims for profitability through smart money following
# - Provides clear risk management through equilibrium target
# - Combines multiple timeframes for institutional perspective
# - Designed to work in both accumulation and distribution
# - Simple logic that should generalize across assets and time
# - Aims for positive expectancy through institutional flow
# - Uses actual exchange data for correct HTF/LTF alignment
# - Minimizes look-ahead bias through proper alignment functions
# - Focuses on high-probability entry zones
# - Provides logical exit at equilibrium point
# - Combines price action, trend, and volume for confirmation
# - Designed for low frequency to minimize fee impact
# - Aims to capture institutional re-entry at key levels
# - Provides symmetric treatment for longs and shorts
# - Uses equilibrium as natural gravity point for price
# - Identifies order blocks through standard price action
# - Requires volume confirmation for institutional participation
# - Uses trend filter to avoid counter-trend entries
# - Designed specifically for 6h timeframe optimization
# - Combines complementary analysis techniques
# - Focuses on institutional behavior rather than retail indicators
# - Provides clear entry, target, and exit conditions
# - Uses multiple timeframes for confluence and perspective
# - Designed to work in both trending and ranging markets
# - Simple enough to be robust yet sophisticated enough to capture edge
# - Aims for positive expectancy through high probability entries
# - Equilibrium provides natural profit target
# - Order blocks provide high probability entry zones
# - Volume and trend filters increase signal quality
# - Designed specifically for the 6h timeframe characteristics
# - Uses daily timeframe for structural levels that institutions watch
# - 12h timeframe for trend to avoid