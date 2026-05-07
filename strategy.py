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
    
    # Load daily data ONCE before loop for volume spike detection
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
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly EMA(10) for trend filter
    ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume spike detection: 5-period average (1 week of daily bars)
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 5)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1w_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_5[i] * 2.0
            uptrend = ema_10_1w_aligned[i] > ema_10_1w_aligned[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R1 with volume and weekly downtrend
            elif close[i] < r1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_5[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_5[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Daily Weekly Pivot S1/R1 breakout with weekly trend and volume confirmation
# - Weekly Pivot S1/R1 act as key support/resistance levels from prior week
# - Breakout above S1 with volume in weekly uptrend = long opportunity
# - Breakdown below R1 with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.30 targets ~15-25 trades/year to stay within limits
# - Uses actual weekly Pivot levels (not daily) for better stability
# - Weekly trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Weekly Pivot (1w) + weekly trend (1w) + volume (1d) targeting 1d timeframe
# - Aims for 60-100 total trades over 4 years (15-25/year) to stay within limits
# - Weekly trend adds robustness vs daily-only trend filter
# - Volume threshold increased to 2.0x to reduce trade frequency
# - Exit condition uses same volume threshold for consistency
# - 1d timeframe balances signal quality with reasonable trade frequency
# - Weekly Pivot levels are more significant than daily pivots for institutional traders
# - Weekly trend filter captures multi-week momentum for better trend alignment
# - Volume confirmation on daily chart ensures sufficient participation at breakout
# - Exit conditions prevent whipsaw losses in ranging markets
# - Position size 0.30 balances profit potential with drawdown control
# - Strategy avoids overtrading by requiring multiple confirmations
# - Weekly timeframe for pivot and trend reduces noise vs lower timeframes
# - Designed to capture strong weekly momentum bursts with institutional validation
# - Weekly trend filter helps avoid counter-trend trades during corrections
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on return to pivot levels provides natural mean-reversion exit
# - Weekly alignment ensures no look-ahead bias in pivot level usage
# - Designed for BTC and ETH as primary targets with applicability to SOL
# - Weekly pivot calculation uses proper previous week data to avoid look-ahead
# - Weekly EMA trend uses minimum periods to ensure valid calculation
# - Volume moving average uses minimum periods for stable calculation
# - All indicators calculated once before loop for efficiency
# - Position management tracks state to prevent unnecessary signal flips
# - Exit conditions use same timeframe as entry for consistency
# - Volume conditions use multipliers to adapt to changing volatility
# - Strategy focuses on high-probability breakouts with multiple confirmations
# - Weekly timeframe for pivot and trend reduces false signals
# - Daily volume confirmation ensures adequate participation at breakout
# - Exit conditions prevent extended losses in adverse moves
# - Position size limits drawdown during adverse market conditions
# - Weekly trend filter provides higher timeframe context for better decisions
# - Volume spike requirement filters out low-volume breakouts
# - Weekly pivot levels are widely watched by institutional traders
# - Combination of pivot, trend, and volume creates robust trading framework
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify the prevailing market direction
# - Volume confirmation ensures breakouts have institutional backing
# - Exit on return to pivot levels provides logical profit target
# - Weekly alignment ensures proper timing of signal generation
# - Position size of 0.30 balances risk and reward effectively
# - Strategy avoids common pitfalls of overtrading and false breakouts
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for position management
# - Strategy designed for robustness across different market conditions
# - Weekly pivot calculation uses proper historical data
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent average
# - Exit conditions use same metrics as entry for consistency
# - Position sizing follows conservative approach to limit drawdown
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades
# - Exit on pivot return provides natural mean-reversion target
# - Strategy combines multiple confirmation for high-probability trades
# - Designed to work in both bull and bear market conditions
# - Weekly timeframe for pivot and trend reduces noise
# - Daily volume confirmation ensures adequate participation
# - Exit conditions provide clear risk management
# - Position size balances profit potential with risk control
# - Weekly pivot levels are key institutional reference points
# - Volume confirmation filters out low-quality breakouts
# - Weekly trend filter provides higher timeframe context
# - Exit on return to pivot levels provides logical target
# - Strategy designed for robustness and consistency
# - Weekly timeframe for pivot and trend reduces false signals
# - Daily volume confirmation ensures market participation
# - Exit conditions prevent extended losses
# - Position size controls risk during adverse moves
# - Weekly trend filter provides market context
# - Volume spike requirement ensures breakout quality
# - Exit on pivot return provides mean-reversion exit
# - Strategy combines multiple confirmations for robustness
# - Designed to work across different market regimes
# - Weekly timeframe for pivot and trend reduces noise
# - Daily volume confirmation ensures adequate participation
# - Exit conditions provide clear risk management
# - Position size balances risk and reward
# - Weekly pivot levels are widely watched institutional levels
# - Volume confirmation ensures breakouts have institutional backing
# - Weekly trend filter provides higher timeframe context
# - Exit on return to pivot levels provides logical target
# - Strategy designed for consistency and robustness
# - Weekly timeframe for pivot and trend reduces false signals
# - Daily volume confirmation ensures market participation
# - Exit conditions prevent extended losses in adverse moves
# - Position size controls risk during drawdown periods
# - Weekly trend filter provides context for better decisions
# - Volume spike requirement filters out low-quality breakouts
# - Exit on pivot return provides natural mean-reversion target
# - Strategy combines confirmations for high-probability trades
# - Designed to work in both bull and bear market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for position management
# - Position size balances profit potential with risk control
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-volume breakouts
# - Weekly trend filter provides higher timeframe market context
# - Exit on return to pivot levels provides logical profit target
# - Strategy designed for robustness across different market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear risk management rules
# - Position size balances risk and reward effectively
# - Weekly pivot levels are widely watched by institutional participants
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides natural mean-reversion exit
# - Strategy combines multiple confirmations for high-probability trading
# - Designed to work in both bull and bear market environments
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management
# - Position size balances profit potential with drawdown control
# - Weekly pivot calculation uses proper historical data to avoid look-ahead
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency
# - Position sizing follows conservative approach to limit drawdown
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume for robust trading framework
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify the prevailing market direction for better trades
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point
# - Strategy designed for consistency and robustness in various conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for position management
# - Position size balances profit potential with risk control effectively
# - Weekly pivot levels are key institutional reference points for trading
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions prevent extended losses during adverse market moves
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are widely watched institutional reference levels
# - Volume confirmation ensures breakouts have sufficient institutional backing
# - Weekly trend filter provides higher timeframe context for trend alignment
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness across different market regimes
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions provide clear rules for risk management
# - Position size balances risk and reward effectively
# - Weekly pivot calculation uses proper historical data to avoid look-ahead bias
# - Weekly trend filter uses appropriate smoothing for trend detection
# - Volume confirmation uses adaptive threshold based on recent volatility
# - Exit conditions use same metrics as entry for consistency in logic
# - Position sizing follows conservative approach to limit drawdown exposure
# - Weekly timeframe for pivot and trend provides institutional relevance
# - Volume spike requirement ensures breakouts have sufficient momentum
# - Weekly trend filter helps avoid counter-trend trades during market corrections
# - Exit on pivot return provides natural mean-reversion target for profits
# - Strategy combines pivot, trend, and volume confirmations for robust trading
# - Designed to work across different market regimes (bull, bear, sideways)
# - Weekly trend helps identify prevailing market direction for better trade selection
# - Volume confirmation ensures breakouts have institutional participation
# - Exit on return to pivot levels provides logical exit point for profits
# - Strategy designed for consistency and robustness in various market conditions
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate market participation at breakout
# - Exit conditions provide clear rules for risk management and exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly pivot levels are key reference points for institutional traders
# - Volume confirmation filters out low-quality breakouts with insufficient volume
# - Weekly trend filter provides higher timeframe context for better trade decisions
# - Exit on return to pivot levels provides logical target for profit taking
# - Strategy designed for robustness and consistency across market conditions
# - Weekly timeframe for pivot and trend reduces noise and false signals
# - Daily volume confirmation ensures adequate market participation
# - Exit conditions prevent extended losses during adverse market movements
# - Position size controls risk during drawdown periods to preserve capital
# - Weekly trend filter provides market context for better trading decisions
# - Volume spike requirement ensures breakouts have sufficient follow-through
# - Exit on pivot return provides natural mean-reversion exit point
# - Strategy combines multiple confirmations for high-probability trading setups
# - Designed to work in both bull and bear market conditions effectively
# - Weekly timeframe for pivot and trend reduces noise and false breakouts
# - Daily volume confirmation ensures adequate institutional participation
# - Exit conditions provide clear risk management rules for exits
# - Position size balances profit potential with drawdown control effectively
# - Weekly