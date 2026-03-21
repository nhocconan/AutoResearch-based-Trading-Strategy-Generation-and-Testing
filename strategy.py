#!/usr/bin/env python3
"""
EXPERIMENT #024 - Weekly Trend + Daily EMA Crossover + RSI Momentum (1d primary)
=====================================================================================
Hypothesis: Daily timeframe captures major crypto moves while avoiding noise of lower TFs.
Using Weekly HMA(21) as the ultimate trend filter ensures we only trade with the macro direction.
Daily EMA(21/55) crossover provides clear entry signals. RSI(14) momentum filter confirms strength.
ADX(14) > 15 ensures we're in a trending environment (lower threshold for daily since ADX moves slower).

Key features:
- Primary TF: 1d (daily candles - fewer but higher quality signals)
- HTF filter: 1w HMA(21) for macro trend direction
- Entry: EMA(21) crossing EMA(55) with momentum confirmation
- Momentum: RSI(14) > 45 for longs, < 55 for shorts (not too strict)
- Trend strength: ADX(14) > 15 (lower threshold for daily data)
- Stoploss: 3.0*ATR(14) trailing (wider for daily volatility)
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit, then trail

Why this should work on daily:
- Weekly HMA filter removes counter-trend trades during major reversals
- EMA crossover on daily captures sustained moves (not noise)
- Looser RSI/ADX thresholds ensure enough trades on daily data
- Conservative 0.25-0.30 sizing controls drawdown during 2022 crash
- 3*ATR stoploss accounts for daily volatility without premature exits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "weekly_trend_daily_ema_rsi_1d_v1"
timeframe = "1d"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1) - Weekly for macro trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Weekly HMA for major trend filter
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate Daily indicators
    ema21 = calculate_ema(close, 21)
    ema55 = calculate_ema(close, 55)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size with strong ADX
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly HMA trend filter (macro direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # Daily EMA crossover signal
        # Fast EMA above slow EMA = bullish
        ema_bullish = ema21[i] > ema55[i]
        ema_bearish = ema21[i] < ema55[i]
        
        # ADX strength filter (lower threshold for daily data)
        adx_strong = adx[i] > 15
        
        # RSI momentum (not too strict to ensure enough trades)
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # DI+ vs DI- for additional trend confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Calculate position size based on ADX strength (dynamic sizing)
        adx_multiplier = min(1.0 + (adx[i] - 15) / 40, 1.15)  # Max 1.15x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Weekly trend up + EMA bullish + ADX strong + RSI momentum + DI+ > DI-
        if (weekly_trend == 1 and ema_bullish and adx_strong and 
            rsi_momentum_long and di_bullish):
            target_signal = position_size
        
        # Short entry: Weekly trend down + EMA bearish + ADX strong + RSI momentum + DI- > DI+
        elif (weekly_trend == -1 and ema_bearish and adx_strong and 
              rsi_momentum_short and di_bearish):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 3.0 * atr[i]  # 3*ATR for daily
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 3*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.0 * entry_atr:  # 2R = 6*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 3.0 * atr[i]  # 3*ATR for daily
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if EMA crossover reverses OR weekly HMA alignment breaks
                ema_reversal_long = ema_bearish
                ema_reversal_short = ema_bullish
                weekly_alignment_broken = (position_side == 1 and weekly_trend == -1) or \
                                          (position_side == -1 and weekly_trend == 1)
                
                if ema_reversal_long or ema_reversal_short or weekly_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals