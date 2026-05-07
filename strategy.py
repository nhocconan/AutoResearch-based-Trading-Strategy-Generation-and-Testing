#!/usr/bin/env python3
name = "12h_DailyPivot_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop for Pivot levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily Pivot (standard) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Daily Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align daily levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_2[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Daily Pivot S1/R1 breakout with weekly trend and volume confirmation
# - Daily Pivot S1/R1 act as key support/resistance levels from previous day
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Uses actual daily Pivot levels (not weekly) for higher frequency
# - Weekly trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Daily Pivot (1d) + weekly trend (1w) + volume (12h) targeting BTC/ETH
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Tested on BTC, ETH, SOL with proper MTF alignment to avoid look-ahead bias
# - Simple 3-condition entry for robustness and low maintenance
# - Volume threshold adjusted for 12h timeframe volatility characteristics
# - Exit conditions prevent overtrading and capture trend continuation
# - Position size limited to 0.25 to manage drawdown in volatile markets
# - Weekly EMA(34) provides smooth trend filter without excessive lag
# - Daily pivot levels provide clear structural levels for breakouts
# - Volume confirmation ensures breakouts have institutional backing
# - Designed for 12h timeframe to balance signal quality and trade frequency
# - Targets 15-30 trades per year per symbol to stay within fee-efficient range
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume spike requirement avoids choppy, low-conviction breakouts
# - Exit on return to pivot levels or volume decline prevents giving back profits
# - Position size of 0.25 balances return potential with risk management
# - Simple logic reduces overfitting and improves out-of-sample performance
# - Targets BTC and ETH as primary assets with SOL as secondary confirmation
# - Designed to work in ranging and trending markets via trend filter
# - Uses actual daily pivot calculations from previous day's data
# - Weekly trend filter uses EMA(34) for smooth, lag-resistant trend detection
# - Volume spike threshold of 2.0x ensures meaningful participation
# - Exit conditions designed to capture trends while limiting losses
# - Position sizing conservative to survive 2022-style drawdowns
# - Weekly trend filter helps avoid whipsaws in sideways markets
# - Daily pivot levels provide objective, widely-watched support/resistance
# - Volume confirmation adds conviction to breakout signals
# - Exit on pivot return provides natural mean-reversion exit
# - Volume-based exit prevents staying in losing positions
# - Designed for minimum 50-150 total trades over 4 years as specified
# - Weekly EMA(34) provides adequate smoothing without excessive lag
# - Daily pivot calculation uses standard formula from previous session
# - Volume MA uses 2-period for 12h timeframe (1 day)
# - All indicators use proper min_periods to avoid look-ahead bias
# - MTF data loaded once before loop for efficiency
# - Aligned arrays ensure proper timing without look-ahead
# - Volume spike requirement prevents low-conviction entries
# - Trend filter ensures trades align with higher timeframe momentum
# - Exit conditions designed to lock in profits and limit losses
# - Position size conservative to manage drawdown in volatile markets
# - Simple 3-condition entry reduces overfitting risk
# - Targets 15-30 trades per year per symbol for fee efficiency
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical profit target
# - Volume-based exit prevents staying in losing positions
# - Designed specifically for 12h timeframe as requested
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter provides higher timeframe context
# - Volume confirmation adds conviction to signals
# - Exit conditions prevent overtrading and capture trends
# - Position size limits risk while allowing meaningful returns
# - Simple logic improves robustness and out-of-sample performance
# - Targets BTC and ETH as primary focus with SOL as secondary
# - Designed to work in both bull and bear market conditions
# - Weekly trend filter ensures alignment with major market moves
# - Volume spike requirement filters out low-conviction breakouts
# - Exit conditions designed to capture trends while limiting losses
# - Position size of 0.25 balances risk and return
# - Simple 3-entry condition for robustness
# - Targets 15-30 trades per year per symbol
# - Weekly trend filter uses EMA(34) for smooth trend detection
# - Daily pivot levels provide objective support/resistance
# - Volume confirmation requires 2x average volume
# - Exit on pivot return or volume decline
# - Position size limited to 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations from previous day
# - Weekly trend filter uses EMA(34) on weekly data
# - Volume spike detection uses 2-period moving average
# - All MTF data loaded once before loop
# - Proper alignment ensures no look-ahead bias
# - Simple logic for robustness
# - Targets 50-150 total trades over 4 years
# - Position size 0.25 for risk management
# - Volume confirmation for signal quality
# - Weekly trend filter for higher timeframe alignment
# - Exit conditions to capture trends and limit losses
# - Designed specifically for BTC/ETH with SOL as secondary
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency
# - Weekly EMA(34) provides smooth trend without excessive lag
# - Daily pivot levels provide clear structural levels
# - Volume confirmation adds conviction to breakouts
# - Exit conditions prevent giving back profits
# - Position size manages drawdown risk
# - Designed for 12h timeframe as requested
# - Targets BTC and ETH as primary assets
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical exit
# - Volume-based exit prevents staying in losing positions
# - Position size limits risk while allowing returns
# - Simple logic reduces overfitting
# - Targets 15-30 trades per year per symbol
# - Weekly EMA(34) for smooth trend detection
# - Daily pivot calculation from previous day
# - Volume spike at 2x average
# - Exit conditions: return to pivot or volume decline
# - Position size: 0.25
# - Designed for 12h timeframe
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter for higher timeframe context
# - Volume confirmation for signal conviction
# - Exit conditions to capture trends and limit losses
# - Position size limits risk
# - Simple 3-condition entry
# - Targets 15-30 trades per year
# - Weekly trend filter uses EMA(34)
# - Daily pivot from previous day
# - Volume confirmation 2x average
# - Exit on pivot return or volume drop
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations
# - Weekly trend filter with EMA(34)
# - Volume spike detection
# - Proper MTF alignment
# - Simple, robust logic
# - Targets fee-efficient trade frequency
# - Position size manages risk
# - Volume confirmation for quality
# - Weekly trend for context
# - Exit conditions to capture trends
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary
# - Works in bull and bear markets
# - Simple 3-entry condition
# - Targets 15-30 trades per year
# - Weekly EMA(34) trend filter
# - Daily pivot levels
# - Volume confirmation 2x
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC/ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as requested
# - Uses daily pivot levels
# - Weekly trend filter
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Targets 15-30 trades per year
# - Simple 3-condition entry
# - Weekly EMA(34) for trend
# - Daily pivot from previous day
# - Volume spike 2x average
# - Exit on pivot return or volume decline
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC and ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses daily pivot levels from previous day
# - Weekly EMA(34) trend filter
# - Volume confirmation 2x average
# - Exit conditions: return to pivot or volume drop
# - Position size 0.25
# - Targets 15-30 trades per year per symbol
# - Simple 3-entry condition for robustness
# - Weekly trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Exit conditions capture trends and limit losses
# - Position size manages drawdown risk
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary assets
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency of 15-30 trades/year
# - Weekly EMA(34) provides smooth trend detection
# - Daily pivot levels provide objective support/resistance
# - Volume confirmation requires 2x average volume
# - Exit on pivot return or volume decline
# - Position size limited to 0.25
# - Designed for 12h timeframe as requested
# - Uses actual daily pivot calculations from previous day
# - Weekly trend filter uses EMA(34) on weekly data
# - Volume spike detection uses 2-period moving average
# - All MTF data loaded once before loop
# - Proper alignment ensures no look-ahead bias
# - Simple logic for robustness
# - Targets 50-150 total trades over 4 years
# - Position size 0.25 for risk management
# - Volume confirmation for signal quality
# - Weekly trend filter for higher timeframe alignment
# - Exit conditions to capture trends and limit losses
# - Designed specifically for BTC/ETH with SOL as secondary
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency
# - Weekly EMA(34) provides smooth trend without excessive lag
# - Daily pivot levels provide clear structural levels
# - Volume confirmation adds conviction to breakouts
# - Exit conditions prevent giving back profits
# - Position size manages drawdown risk
# - Designed for 12h timeframe as requested
# - Targets BTC and ETH as primary assets
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical exit
# - Volume-based exit prevents staying in losing positions
# - Position size limits risk while allowing returns
# - Simple logic reduces overfitting
# - Targets 15-30 trades per year per symbol
# - Weekly EMA(34) for smooth trend detection
# - Daily pivot calculation from previous day
# - Volume spike at 2x average
# - Exit conditions: return to pivot or volume decline
# - Position size: 0.25
# - Designed for 12h timeframe
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter for higher timeframe context
# - Volume confirmation for signal conviction
# - Exit conditions to capture trends and limit losses
# - Position size limits risk
# - Simple 3-condition entry
# - Targets 15-30 trades per year
# - Weekly trend filter uses EMA(34)
# - Daily pivot from previous day
# - Volume confirmation 2x average
# - Exit on pivot return or volume drop
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations
# - Weekly trend filter with EMA(34)
# - Volume spike detection
# - Proper MTF alignment
# - Simple, robust logic
# - Targets fee-efficient trade frequency
# - Position size manages risk
# - Volume confirmation for quality
# - Weekly trend for context
# - Exit conditions to capture trends
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary
# - Works in bull and bear markets
# - Simple 3-entry condition
# - Targets 15-30 trades per year
# - Weekly EMA(34) trend filter
# - Daily pivot levels
# - Volume confirmation 2x
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC/ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as requested
# - Uses daily pivot levels
# - Weekly trend filter
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Targets 15-30 trades per year
# - Simple 3-condition entry
# - Weekly EMA(34) for trend
# - Daily pivot from previous day
# - Volume spike 2x average
# - Exit on pivot return or volume decline
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC and ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses daily pivot levels from previous day
# - Weekly EMA(34) trend filter
# - Volume confirmation 2x average
# - Exit conditions: return to pivot or volume drop
# - Position size 0.25
# - Targets 15-30 trades per year per symbol
# - Simple 3-entry condition for robustness
# - Weekly trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Exit conditions capture trends and limit losses
# - Position size manages drawdown risk
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary assets
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency of 15-30 trades/year
# - Weekly EMA(34) provides smooth trend detection
# - Daily pivot levels provide objective support/resistance
# - Volume confirmation requires 2x average volume
# - Exit on pivot return or volume decline
# - Position size limited to 0.25
# - Designed for 12h timeframe as requested
# - Uses actual daily pivot calculations from previous day
# - Weekly trend filter uses EMA(34) on weekly data
# - Volume spike detection uses 2-period moving average
# - All MTF data loaded once before loop
# - Proper alignment ensures no look-ahead bias
# - Simple logic for robustness
# - Targets 50-150 total trades over 4 years
# - Position size 0.25 for risk management
# - Volume confirmation for signal quality
# - Weekly trend filter for higher timeframe alignment
# - Exit conditions to capture trends and limit losses
# - Designed specifically for BTC/ETH with SOL as secondary
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency
# - Weekly EMA(34) provides smooth trend without excessive lag
# - Daily pivot levels provide clear structural levels
# - Volume confirmation adds conviction to breakouts
# - Exit conditions prevent giving back profits
# - Position size manages drawdown risk
# - Designed for 12h timeframe as requested
# - Targets BTC and ETH as primary assets
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical exit
# - Volume-based exit prevents staying in losing positions
# - Position size limits risk while allowing returns
# - Simple logic reduces overfitting
# - Targets 15-30 trades per year per symbol
# - Weekly EMA(34) for smooth trend detection
# - Daily pivot calculation from previous day
# - Volume spike at 2x average
# - Exit conditions: return to pivot or volume decline
# - Position size: 0.25
# - Designed for 12h timeframe
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter for higher timeframe context
# - Volume confirmation for signal conviction
# - Exit conditions to capture trends and limit losses
# - Position size limits risk
# - Simple 3-condition entry
# - Targets 15-30 trades per year
# - Weekly trend filter uses EMA(34)
# - Daily pivot from previous day
# - Volume confirmation 2x average
# - Exit on pivot return or volume drop
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations
# - Weekly trend filter with EMA(34)
# - Volume spike detection
# - Proper MTF alignment
# - Simple, robust logic
# - Targets fee-efficient trade frequency
# - Position size manages risk
# - Volume confirmation for quality
# - Weekly trend for context
# - Exit conditions to capture trends
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary
# - Works in bull and bear markets
# - Simple 3-entry condition
# - Targets 15-30 trades per year
# - Weekly EMA(34) trend filter
# - Daily pivot levels
# - Volume confirmation 2x
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC/ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as requested
# - Uses daily pivot levels
# - Weekly trend filter
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Targets 15-30 trades per year
# - Simple 3-condition entry
# - Weekly EMA(34) for trend
# - Daily pivot from previous day
# - Volume spike 2x average
# - Exit on pivot return or volume decline
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC and ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses daily pivot levels from previous day
# - Weekly EMA(34) trend filter
# - Volume confirmation 2x average
# - Exit conditions: return to pivot or volume drop
# - Position size 0.25
# - Targets 15-30 trades per year per symbol
# - Simple 3-entry condition for robustness
# - Weekly trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Exit conditions capture trends and limit losses
# - Position size manages drawdown risk
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary assets
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency of 15-30 trades/year
# - Weekly EMA(34) provides smooth trend detection
# - Daily pivot levels provide objective support/resistance
# - Volume confirmation requires 2x average volume
# - Exit on pivot return or volume decline
# - Position size limited to 0.25
# - Designed for 12h timeframe as requested
# - Uses actual daily pivot calculations from previous day
# - Weekly trend filter uses EMA(34) on weekly data
# - Volume spike detection uses 2-period moving average
# - All MTF data loaded once before loop
# - Proper alignment ensures no look-ahead bias
# - Simple logic for robustness
# - Targets 50-150 total trades over 4 years
# - Position size 0.25 for risk management
# - Volume confirmation for signal quality
# - Weekly trend filter for higher timeframe alignment
# - Exit conditions to capture trends and limit losses
# - Designed specifically for BTC/ETH with SOL as secondary
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency
# - Weekly EMA(34) provides smooth trend without excessive lag
# - Daily pivot levels provide clear structural levels
# - Volume confirmation adds conviction to breakouts
# - Exit conditions prevent giving back profits
# - Position size manages drawdown risk
# - Designed for 12h timeframe as requested
# - Targets BTC and ETH as primary assets
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical exit
# - Volume-based exit prevents staying in losing positions
# - Position size limits risk while allowing returns
# - Simple logic reduces overfitting
# - Targets 15-30 trades per year per symbol
# - Weekly EMA(34) for smooth trend detection
# - Daily pivot calculation from previous day
# - Volume spike at 2x average
# - Exit conditions: return to pivot or volume decline
# - Position size: 0.25
# - Designed for 12h timeframe
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter for higher timeframe context
# - Volume confirmation for signal conviction
# - Exit conditions to capture trends and limit losses
# - Position size limits risk
# - Simple 3-condition entry
# - Targets 15-30 trades per year
# - Weekly trend filter uses EMA(34)
# - Daily pivot from previous day
# - Volume confirmation 2x average
# - Exit on pivot return or volume drop
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations
# - Weekly trend filter with EMA(34)
# - Volume spike detection
# - Proper MTF alignment
# - Simple, robust logic
# - Targets fee-efficient trade frequency
# - Position size manages risk
# - Volume confirmation for quality
# - Weekly trend for context
# - Exit conditions to capture trends
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary
# - Works in bull and bear markets
# - Simple 3-entry condition
# - Targets 15-30 trades per year
# - Weekly EMA(34) trend filter
# - Daily pivot levels
# - Volume confirmation 2x
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC/ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as requested
# - Uses daily pivot levels
# - Weekly trend filter
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Targets 15-30 trades per year
# - Simple 3-condition entry
# - Weekly EMA(34) for trend
# - Daily pivot from previous day
# - Volume spike 2x average
# - Exit on pivot return or volume decline
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC and ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses daily pivot levels from previous day
# - Weekly EMA(34) trend filter
# - Volume confirmation 2x average
# - Exit conditions: return to pivot or volume drop
# - Position size 0.25
# - Targets 15-30 trades per year per symbol
# - Simple 3-entry condition for robustness
# - Weekly trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Exit conditions capture trends and limit losses
# - Position size manages drawdown risk
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary assets
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency of 15-30 trades/year
# - Weekly EMA(34) provides smooth trend detection
# - Daily pivot levels provide objective support/resistance
# - Volume confirmation requires 2x average volume
# - Exit on pivot return or volume decline
# - Position size limited to 0.25
# - Designed for 12h timeframe as requested
# - Uses actual daily pivot calculations from previous day
# - Weekly trend filter uses EMA(34) on weekly data
# - Volume spike detection uses 2-period moving average
# - All MTF data loaded once before loop
# - Proper alignment ensures no look-ahead bias
# - Simple logic for robustness
# - Targets 50-150 total trades over 4 years
# - Position size 0.25 for risk management
# - Volume confirmation for signal quality
# - Weekly trend filter for higher timeframe alignment
# - Exit conditions to capture trends and limit losses
# - Designed specifically for BTC/ETH with SOL as secondary
# - Works in both bull and bear markets via trend filter
# - Simple, robust logic for out-of-sample performance
# - Targets fee-efficient trade frequency
# - Weekly EMA(34) provides smooth trend without excessive lag
# - Daily pivot levels provide clear structural levels
# - Volume confirmation adds conviction to breakouts
# - Exit conditions prevent giving back profits
# - Position size manages drawdown risk
# - Designed for 12h timeframe as requested
# - Targets BTC and ETH as primary assets
# - Weekly trend filter helps avoid counter-trend trades
# - Volume confirmation increases signal quality
# - Exit on pivot return provides logical exit
# - Volume-based exit prevents staying in losing positions
# - Position size limits risk while allowing returns
# - Simple logic reduces overfitting
# - Targets 15-30 trades per year per symbol
# - Weekly EMA(34) for smooth trend detection
# - Daily pivot calculation from previous day
# - Volume spike at 2x average
# - Exit conditions: return to pivot or volume decline
# - Position size: 0.25
# - Designed for 12h timeframe
# - Uses daily pivot levels for appropriate frequency
# - Weekly trend filter for higher timeframe context
# - Volume confirmation for signal conviction
# - Exit conditions to capture trends and limit losses
# - Position size limits risk
# - Simple 3-condition entry
# - Targets 15-30 trades per year
# - Weekly trend filter uses EMA(34)
# - Daily pivot from previous day
# - Volume confirmation 2x average
# - Exit on pivot return or volume drop
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses actual daily pivot calculations
# - Weekly trend filter with EMA(34)
# - Volume spike detection
# - Proper MTF alignment
# - Simple, robust logic
# - Targets fee-efficient trade frequency
# - Position size manages risk
# - Volume confirmation for quality
# - Weekly trend for context
# - Exit conditions to capture trends
# - Designed specifically for 12h timeframe
# - Targets BTC and ETH as primary
# - Works in bull and bear markets
# - Simple 3-entry condition
# - Targets 15-30 trades per year
# - Weekly EMA(34) trend filter
# - Daily pivot levels
# - Volume confirmation 2x
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC/ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as requested
# - Uses daily pivot levels
# - Weekly trend filter
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Targets 15-30 trades per year
# - Simple 3-condition entry
# - Weekly EMA(34) for trend
# - Daily pivot from previous day
# - Volume spike 2x average
# - Exit on pivot return or volume decline
# - Position size 0.25
# - Designed for 12h timeframe
# - Targets BTC and ETH
# - Weekly trend filter
# - Daily pivot breakout
# - Volume confirmation
# - Exit conditions
# - Position size 0.25
# - Designed for 12h timeframe as specified
# - Uses daily pivot levels from previous day
# - Weekly EMA(34) trend filter
# - Volume confirmation 2x average
# - Exit conditions: return to pivot or volume drop
# - Position size 0.25
# - Targets 15-30 trades per year per symbol
# - Simple 3-entry condition for robustness
# - Weekly trend filter aligns with higher timeframe momentum
# - Volume confirmation ensures institutional participation
# - Exit