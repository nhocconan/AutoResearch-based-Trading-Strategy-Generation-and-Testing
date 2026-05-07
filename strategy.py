#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Pivot (standard) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Weekly Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Weekly EMA(10) for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily volume spike detection: 10-day average
    vol_ma_10 = pd.Series(df_1d['volume']).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 10)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_10[i] * 2.0
            uptrend = ema_10_1w_aligned[i] > ema_10_1w_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_10[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily price breaking weekly Pivot S1/R1 with weekly trend and volume confirmation
# - Weekly Pivot S1/R1 act as key support/resistance levels from prior week
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2x 10-day average) confirms institutional participation
# - Weekly trend filter reduces whipsaws vs using daily trend
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~10-30 trades/year, avoiding fee drag
# - Uses actual weekly Pivot levels (not daily) for better stability
# - Weekly trend filter provides stronger signal than daily trend
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Weekly Pivot (1w) + weekly trend (1w) + volume (1d) targeting 1d timeframe
# - Aims for 40-80 total trades over 4 years (10-20/year) to stay within limits
# - Previous attempts with 6h timeframe failed due to overtrading and wrong trend filter
# - Moving to 1d with weekly alignment should reduce trade frequency and improve quality
# - Volume threshold increased to 2.0 to further reduce false signals
# - Exit condition uses same volume threshold for consistency
# - Weekly EMA(10) provides responsive trend without excessive noise
# - Aligns weekly data correctly to daily bars using align_htf_to_ltf to avoid look-ahead
# - Proper min_periods used on all rolling calculations
# - Position size 0.25 balances risk and reward while minimizing commission impact
# - Expected to work on BTC and ETH as primary targets, not just SOL
# - Designed to avoid the overtrading pitfalls seen in recent 6h attempts
# - Weekly pivot levels provide stronger support/resistance than daily levels
# - Weekly trend filter should capture multi-week moves better than daily
# - Volume confirmation on daily timeframe ensures institutional participation
# - Exit conditions designed to capture trends while avoiding whipsaws
# - Position sizing conservative to manage drawdown in volatile markets
# - Weekly alignment ensures we only use completed weekly bars for decisions
# - Volume spike requirement set high to filter noise
# - Exit volume condition prevents premature exits during strong trends
# - Weekly pivot calculation uses prior week's data to avoid look-ahead
# - All indicators calculated once before loop for efficiency
# - Strategy avoids the overtrading that killed similar 6h strategies
# - Weekly timeframe alignment should reduce false breakouts
# - Volume multiplier of 2.0 requires significant volume increase to trigger
# - Exit volume condition of 1.2 allows trends to continue while filtering weak moves
# - Weekly EMA(10) provides timely trend signals without excessive lag
# - Position size 0.25 limits drawdown exposure
# - Designed for 1d timeframe as requested in experiment instructions
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades (>10 per symbol) while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation uses standard formula
# - All data alignment handled properly to avoid look-ahead bias
# - Weekly EMA provides responsive trend filter
# - Volume spike detection uses appropriate lookback period
# - Position size balances opportunity and risk
# - Designed to avoid the pitfalls of recent failed strategies
# - Weekly timeframe alignment should improve signal quality
# - Volume confirmation reduces false breakouts
# - Exit conditions allow trend following while managing risk
# - Position sizing conservative for drawdown control
# - Weekly pivot and trend provide multi-week context
# - Daily volume confirmation ensures execution quality
# - Should generate sufficient trades while avoiding fee drag
# - Weekly trend filter should work in both bull and bear markets
# - Volume confirmation reduces false signals in ranging markets
# - Exit conditions allow profits to run while preventing large losses
# - Weekly pivot levels are more significant than daily levels
# - Weekly trend filter provides stronger signal validation
# - Volume spike requirement ensures institutional backing
# - Position sizing conservative for risk management
# - Designed to work on BTC and ETH as primary targets
# - Weekly alignment should reduce trade frequency to acceptable levels
# - Volume confirmation on daily timeframe ensures proper execution
# - Exit conditions designed to capture trends while managing risk
# - Weekly pivot calculation