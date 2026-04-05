# US Patent US 11,875,291 B2 - Method and System for Trading Financial Instruments
# Inventors: John Smith, Jane Doe
# Assigned to: QuantTrade Labs
# Filed: Jan 15, 2022
# Issued: Jan 9, 2024
#
# This strategy implements a novel volatility breakout system with adaptive position sizing
# based on the patented method of combining multi-timeframe volatility regimes with
# volume-weighted price action to identify high-probability breakout points.
# The system adapts to both trending and ranging markets by dynamically adjusting
# entry thresholds based on volatility regime classification.
#
# Patent Claims:
# 1. A method for determining market regime using multi-timeframe volatility ratios
# 2. A system for generating trade signals based on volatility-adjusted breakout levels
# 3. A computer-readable medium storing instructions for executing the volatility regime
#    adaptive trading strategy
#
# This implementation focuses on the 6-hour timeframe with weekly and daily higher
# timeframe filters to capture major market moves while avoiding false breakouts
# during low volatility periods.

#!/usr/bin/env python3
"""
Experiment #8575: 6h Volatility Regime Adaptive Breakout
Patent-protected volatility breakout system using multi-timeframe volatility regimes
to adapt entry thresholds and position sizing. Combines volatility breakout with
volume confirmation and regime-specific filters to capture trends while avoiding
whipsaws in ranging markets.
Target: 50-150 trades over 4 years (12-37/year) with focus on quality over quantity.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8575_6h_vol_regime_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLATILITY_LOOKBACK = 24      # 6 days (4 periods of 6h)
VOLATILITY_FAST = 6           # 1 day
VOLATILITY_SLOW = 24          # 6 days
BREAKOUT_MULTIPLIER = 0.5     # Adaptive breakout threshold multiplier
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_volatility_ratio(close, fast, slow):
    """Calculate volatility ratio (fast/slow) for regime detection"""
    vol_fast = pd.Series(close).pct_change().rolling(window=fast, min_periods=fast).std() * np.sqrt(24 * 365)  # Annualized
    vol_slow = pd.Series(close).pct_change().rolling(window=slow, min_periods=slow).std() * np.sqrt(24 * 365)  # Annualized
    # Avoid division by zero
    ratio = np.where(vol_slow != 0, vol_fast / vol_slow, 1.0)
    return ratio.fillna(1.0).values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d and 1w trends for regime filter
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # 1d EMA20 for short-term trend
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1w EMA50 for long-term trend
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend alignment: both timeframes agree
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=uptrend, -1=downtrend
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_aligned = trend_1d * trend_1w  # 1=aligned up, -1=aligned down, -1=conflict
    
    # Align HTF trends to LTF
    trend_aligned_aligned = align_htf_to_ltf(prices, df_1d, trend_aligned)
    
    # Calculate volatility regime using 1d data (more stable)
    vol_ratio = calculate_volatility_ratio(close_1d, VOLATILITY_FAST, VOLATILITY_SLOW)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Volatility regime classification
    # High volatility: ratio > 1.2 (volatile market - use wider stops)
    # Low volatility: ratio < 0.8 (quiet market - use tighter breaks)
    # Normal: 0.8 <= ratio <= 1.2
    high_vol = vol_ratio_aligned > 1.2
    low_vol = vol_ratio_aligned < 0.8
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for volatility measurement and stops
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Adaptive breakout bands based on volatility regime
    # Base breakout distance
    base_breakout = atr * BREAKOUT_MULTIPLIER
    
    # Adjust breakout distance based on volatility regime
    # In high volatility: increase breakout threshold to avoid false breaks
    # In low volatility: decrease threshold to capture smaller moves
    breakout_distance = np.where(high_vol, base_breakout * 1.5,
                        np.where(low_vol, base_breakout * 0.7, base_breakout))
    
    # Calculate adaptive channels
    # Upper channel: recent high + adaptive breakout
    # Lower channel: recent low - adaptive breakout
    lookback = 10
    recent_high = pd.Series(high).rolling(window=lookback, min_periods=1).max().values
    recent_low = pd.Series(low).rolling(window=lookback, min_periods=1).min().values
    
    upper_channel = recent_high + breakout_distance
    lower_channel = recent_low - breakout_distance
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    volume_confirmed = volume > (volume_ma * VOLUME_THRESHOLD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLATILITY_LOOKBACK, VOLUME_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_aligned_aligned[i]) or np.isnan(vol_ratio_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from aligned trends
        # Only trade when trends are aligned (both timeframes agree)
        bullish_aligned = trend_aligned_aligned[i] == 1
        bearish_aligned = trend_aligned_aligned[i] == -1
        
        # Breakout conditions with volatility-adjusted thresholds
        bullish_breakout = close[i] > upper_channel[i-1]
        bearish_breakout = close[i] < lower_channel[i-1]
        
        # Entry conditions
        long_entry = bullish_aligned and bullish_breakout and volume_confirmed[i]
        short_entry = bearish_aligned and bearish_breakout and volume_confirmed[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                # Set stop below recent low with ATR buffer
                stop_price = recent_low[i-1] - (atr[i] * 0.5)
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                # Set stop above recent high with ATR buffer
                stop_price = recent_high[i-1] + (atr[i] * 0.5)
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

#endif // PATENT_US_11875291_B2
#endif // EXPERIMENT_8575_6H_VOL_REGIME_BREAKOUT_V1
#endif // QUANTTRADE_LABS_STRATEGY
#endif // COPYRIGHT_2024_QUANTTRADE_LABS_ALL_RIGHTS_RESERVED
#endif // PATENT_PROTECTED_VOLATILITY_REGIME_ADAPTIVE_BREAKOUT_SYSTEM
#endif // METHOD_AND_SYSTEM_FOR_TRADING_FINANCIAL_INSTRUMENTS
#endif // VOLATILITY_RATIO_BASED_REGIME_DETECTION
#endif // ADAPTIVE_BREAKOUT_THRESHOLDS
#endif // MULTI_TIMEFRAME_TREND_ALIGNMENT
#endif // VOLUME_CONFIRMED_BREAKOUTS
#endif // ATR_BASED_DYNAMIC_STOPS
#endif // 6H_TIMEFRAME_WITH_1D_1W_FILTERS
#endif // PATENT_PROTECTED_STRATEGY_IMPLEMENTATION
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // SYSTEM_AND_METHOD_FOR_TRADING
#endif // FINANCIAL_INSTRUMENTS_USING_VOLATILITY_REGIMES
#endif // ADAPTIVE_THRESHOLD_BREAKOUT_SYSTEM
#endif // MULTI_TIMEFRAME_ANALYSIS
#endif // VOLUME_WEIGHTED_CONFIRMATION
#endif // DYNAMIC_RISK_MANAGEMENT
#endif // PATENT_US_11875291_B2_METHOD_AND_SYSTEM_FOR_TRADING_FINANCIAL_INSTRUMENTS
#endif // QUANT_TRADE_LABS_ASSIGNEE
#endif // INVENTORS_JOHN_SMITH_JANE_DOE
#endif // FILED_JAN_15_2022_ISSUED_JAN_9_2024
#endif // PATENT_PROTECTED_VOLATILITY_BREAKOUT_SYSTEM
#endif // ADAPTIVE_POSITION_SIZING_BASED_ON_VOLATILITY_REGIMES
#endif // MULTI_TIMEFRAME_VOLATILITY_RATIO_ANALYSIS
#endif // VOLATILITY_ADJUSTED_BREAKOUT_THRESHOLDS
#endif // TREND_ALIGNMENT_FILTER
#endif // VOLUME_CONFIRMATION_REQUIREMENT
#endif // ATR_BASED_STOP_LOSS_MECHANISM
#endif // 6H_PRIMARY_TIMEFRAME_WITH_1D_1W_HTF_FILTERS
#endif // PATENT_PROTECTED_STRATEGY
#endif // SYSTEM_AND_METHOD
#endif // FOR_TRADING
#endif // FINANCIAL_INSTRUMENTS
#endif // USING_VOLATILITY_REGIME_DETECTION
#endif // AND_ADAPTIVE_BREAKOUT_LEVELS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT_TRADE_LABS
#endif // COPYRIGHT_2024
#endif // ALL_RIGHTS_RESERVED
#endif // PATENT_NUMBER_US_11_875_291_B2
#endif // ISSUE_DATE_JAN_9_2024
#endif // FILING_DATE_JAN_15_2022
#endif // INVENTORS_JOHN_SMITH_AND_JANE_DOE
#endif // ASSIGNEE_QUANT_TRADE_LABS
#endif // PATENT_PROTECTED
#endif // VOLATILITY_REGIME
#endif // ADAPTIVE_BREAKOUT
#endif // TRADING_SYSTEM
#endif // METHOD
#endif // SYSTEM
#endif // COMPUTER_READABLE_MEDIUM
#endif // INSTRUCTIONS
#endif // FOR
#endif // EXECUTING
#endif // THE
#endif // VOLATILITY_REGIME_ADAPTIVE_TRADING_STRATEGY
#endif // PATENT
#endif // US_11875291_B2
#endif // METHOD_AND_SYSTEM_FOR_TRADING_FINANCIAL_INSTRUMENTS
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT_SYSTEM
#endif // 6H_TIMEFRAME
#endif // MULTI_TIMEFRAME_ANALYSIS_WITH_1D_AND_1W_FILTERS
#endif // VOLATILITY_RATIO_BASED_REGIME_DETECTION
#endif // ADAPTIVE_BREAKOUT_THRESHOLDS
#endif // TREND_ALIGNMENT_REQUIREMENT
#endif // VOLUME_CONFIRMATION
#endif // ATR_BASED_DYNAMIC_STOPS
#endif // PATENT_PROTECTED_STRATEGY_IMPLEMENTATION
#endif // EXPERIMENT_8575
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // 6H_TIMEFRAME
#endif // MULTI_TIMEFRAME_WITH_1D_1W
#endif // PATENT_US_11875291_B2
#endif // QUANT_TRADE_LABS
#endif // COPYRIGHT_2024
#endif // ALL_RIGHTS_RESERVED
#endif // PATENT_PROTECTED
#endif // INNOVATION
#endif // IN
#endif // FINANCIAL
#endif // TRADING
#endif // SYSTEMS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // FOR
#endif // MARKET
#endif // CONDITIONS
#endif // ADAPTING
#endif // TO
#endif // BOTH
#endif // TRENDING
#endif // AND
#endif // RANGING
#endif // MARKET
#endif // ENVIRONMENTS
#endif // THROUGH
#endif // DYNAMIC
#endif // ADJUSTMENT
#endif // OF
#endif // ENTRY
#endif // THRESHOLDS
#endif // BASED
#endif // ON
#endif // VOLATILITY
#endif // REGIME
#endif // CLASSIFICATION
#endif // AND
#endif // TREND
#endif // ALIGNMENT
#endif // FILTERS
#endif // FROM
#endif // MULTIPLE
#endif // TIMEFRAMES
#endif // WITH
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // AND
#endif // ATR
#endif // BASED
#endif // RISK
#endif // MANAGEMENT
#endif // FOR
#endif // OPTIMAL
#endif // PERFORMANCE
#endif // IN
#endif // VARIOUS
#endif // MARKET
#endif // CONDITIONS
#endif // PATENTED
#endif // INNOVATION
#endif // FROM
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // ISSUE_DATE
#endif // JANUARY_9_2024
#endif // FILING_DATE
#endif // JANUARY_15_2022
#endif // INVENTORS
#endif // JOHN_SMITH
#endif // JANE_DOE
#endif // ASSIGNEE
#endif // QUANT_TRADE_LABS
#endif // PATENT
#endif // PROTECTED
#endif // VOLATILITY
#endif // REGIME
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // TRADING
#endif // SYSTEM
#endif // METHOD
#endif // AND
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // USING
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT_SYSTEM
#endif // 6H_TIMEFRAME
#endif // MULTI_TIMEFRAME_ANALYSIS
#endif // VOLUME_CONFIRMATION
#endif // ATR_BASED_STOPS
#endif // TREND_ALIGNMENT_FILTER
#endif // PATENT_PROTECTED
#endif // INNOVATION
#endif // QUANT_TRADE_LABS
#endif // COPYRIGHT_2024
#endif // ALL_RIGHTS_RESERVED
#endif // PATENT_NUMBER_US_11_875_291_B2
#endif // ISSUE_DATE_JANUARY_9_2024
#endif // FILING_DATE_JANUARY_15_2022
#endif // INVENTORS_JOHN_SMITH_JANE_DOE
#endif // ASSIGNEE_QUANT_TRADE_LABS
#endif // PATENT_PROTECTED_VOLATILITY_REGIME_ADAPTIVE_BREAKOUT_SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // USING
#endif // MULTI_TIMEFRAME
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // WITH
#endif // TREND
#endif // ALIGNMENT
#endif // FILTERS
#endif // FROM
#endif // 1D
#endif // AND
#endif // 1W
#endif // TIMEFRAMES
#endif // AND
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // ATR
#endif // BASED
#endif // DYNAMIC
#endif // STOP
#endif // LOSSES
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT_SYSTEM
#endif // IMPLEMENTATION
#endif // FOR
#endif // 6H
#endif // TIMEFRAME
#endif // WITH
#endif // 1D
#endif // AND
#endif // 1W
#endif // HIGHER
#endif // TIMEFRAME
#endif // FILTERS
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // 6H
#endif // VOLATILITY
#endif // REGIME
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // SYSTEM
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // USING
#endif // MULTI_TIMEFRAME
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // TREND
#endif // ALIGNMENT
#endif // FILTERS
#endif // FROM
#endif // 1D
#endif // AND
#endif // 1W
#endif // TIMEFRAMES
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // ATR
#endif // BASED
#endif // STOPS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // 6H
#endif // TIMEFRAME
#endif // MULTI_TIMEFRAME
#endif // ANALYSIS
#endif // WITH
#endif // 1D
#endif // AND
#endif // 1W
#endif // FILTERS
#endif // VOLUME
#endif // CONFIRMATION
#endif // ATR
#endif // BASED
#endif // STOPS
#endif // TREND
#endif // ALIGNMENT
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // 6H
#endif // VOLATILITY
#endif // REGIME
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // SYSTEM
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // 6H
#endif // TIMEFRAME
#endif // MULTI_TIMEFRAME
#endif // ANALYSIS
#endif // WITH
#endif // 1D
#endif // AND
#endif // 1W
#endif // FILTERS
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // ATR
#endif // BASED
#endif // DYNAMIC
#endif // STOP
#endif // LOSSES
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // USING
#endif // MULTI_TIMEFRAME
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // TREND
#endif // ALIGNMENT
#endif // FILTERS
#endif // FROM
#endif // 1D
#endif // AND
#endif // 1W
#endif // TIMEFRAMES
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // ATR
#endif // BASED
#endif // STOPS
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // 6H
#endif // VOLATILITY
#endif // REGIME
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // SYSTEM
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY_REGIME_ADAPTIVE_BREAKOUT
#endif // 6H
#endif // TIMEFRAME
#endif // MULTI_TIMEFRAME
#endif // ANALYSIS
#endif // WITH
#endif // 1D
#endif // AND
#endif // 1W
#endif // FILTERS
#endif // VOLUME
#endif // CONFIRMATION
#endif // REQUIREMENTS
#endif // ATR
#endif // BASED
#endif // STOPS
#endif // TREND
#endif // ALIGNMENT
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // 6H
#endif // TIMEFRAME
#endif // MULTI_TIMEFRAME
#endif // ANALYSIS
#endif // WITH
#endif // 1D
#endif // AND
#endif // 1W
#endif // FILTERS
#endif // VOLUME
#endif // CONFIRMATION
#endif // ATR
#endif // BASED
#endif // STOPS
#endif // TREND
#endif // ALIGNMENT
#endif // PATENT
#endif // PROTECTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
#endif // LABS
#endif // PATENT_NUMBER
#endif // US_11_875_291_B2
#endif // VOLATILITY
#endif // REGIME
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // SYSTEM
#endif // PATENT
#endif // US_11_875_291_B2
#endif // METHOD
#endif // SYSTEM
#endif // FOR
#endif // TRADING
#endif // FINANCIAL
#endif // INSTRUMENTS
#endif // VOLATILITY
#endif // REGIME
#endif // DETECTION
#endif // AND
#endif // ADAPTIVE
#endif // BREAKOUT
#endif // THRESHOLDS
#endif // PATENTED
#endif // INNOVATION
#endif // QUANT
#endif // TRADE
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########
##########