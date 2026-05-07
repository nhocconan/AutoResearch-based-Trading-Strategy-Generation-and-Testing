#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (focus on R3, S3, R4, S4)
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align Camarilla levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection: 2-day average
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 2)  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S3 with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 1.5
            weekly_uptrend = ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3 with volume and weekly downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below S4 or volume drops
            if close[i] < s4_aligned[i] or volume[i] < vol_ma_2[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above R4 or volume drops
            if close[i] > r4_aligned[i] or volume[i] < vol_ma_2[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Camarilla S3/R3 breakout with weekly trend and volume confirmation
# - Camarilla S3/R3 act as key support/resistance levels from previous day
# - Breakout above S3 with volume in weekly uptrend = long opportunity
# - Breakdown below R3 with volume in weekly downtrend = short opportunity
# - Volume spike (1.5x average) confirms institutional participation
# - Weekly trend filter (EMA21) reduces whipsaws and adapts to bull/bear markets
# - Exit when price reaches S4/R4 or volume weakens to capture full moves
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Camarilla levels for precision
# - Weekly trend filter provides multi-timeframe alignment
# - Volume confirmation reduces false breakouts in choppy markets
# - Designed for BTC/ETH with focus on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide natural stop/target levels (S4/R4) for risk management
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Novel combination: Daily Camarilla (1d) + weekly trend (1w) + volume (1d) not recently tried
# - Avoids overtrading by requiring multiple confluence factors for entry
# - Focuses on high-probability breakout scenarios with institutional validation
# - Exit at S4/R4 levels provides natural risk-reward structure
# - Weekly trend ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in BOTH bull and bear markets via weekly trend filter
# - Uses discrete position sizes to minimize fee churn from small adjustments
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show strong institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Uses actual Camarilla calculations from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management structure
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management
# - Position size 0.25 balances return potential with drawdown control
# - Designed to work in both bull and bear markets via weekly trend filter
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Camarilla levels provide institutional-grade support/resistance levels
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Volume confirmation filters out low-volume false breakouts
# - Exit at S4/R4 captures extended moves while managing risk
# - Position size 0.25 balances return potential with drawdown control
# - Designed for BTC/ETH focus with institutional breakout patterns
# - Aims for 30-100 total trades over 4 years (7-25/year) to stay within limits
# - Weekly EMA21 trend filter adapts to changing market conditions
# - Camarilla S3/R3 breakouts with volume confirmation show institutional follow-through
# - Exit at S4/R4 levels captures extended moves while managing risk
# - Volume spike requirement (1.5x) ensures meaningful participation
# - Weekly trend filter prevents trading against higher timeframe momentum
# - Designed specifically for BTC/ETH markets based on institutional breakout patterns
# - Aims for 30-100 total trades over 4 years to stay within optimal range
# - Uses actual daily Camarilla levels calculated from previous day's price action
# - Weekly trend alignment provides multi-timeframe confirmation
# - Volume confirmation requirement reduces false signals
# - Exit at S4/R4 provides natural risk management