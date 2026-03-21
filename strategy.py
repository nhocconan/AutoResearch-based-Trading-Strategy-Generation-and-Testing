#!/usr/bin/env python3
"""
EXPERIMENT #032 - HMA Trend + RSI Pullback + Volatility Filter (30m primary, 4h HTF)
====================================================================================
Hypothesis: 30m timeframe offers better entry timing than 4h while avoiding noise of 15m.
4h HMA(21) provides robust trend filter (proven in current best strategy). 
RSI(14) pullback to 45-55 zone captures mean reversion within trend. 
ATR percentile filter avoids trading during extreme volatility spikes (common cause of DD).
Donchian(20) breakout confirmation reduces false signals.

Key differences from failed attempts:
- Simpler indicator stack (less overfitting than KAMA+MACD+ADX+Volume combos)
- Volatility regime filter (ATR percentile) - avoids trading during chaos
- Conservative position sizing (0.25 base, max 0.30)
- Tighter stoploss (2*ATR) with proper trailing
- Discrete signal levels to minimize fee churn

Position sizing: 0.25 base, 0.30 max on strong signals
Stoploss: 2*ATR trailing, signal→0 when hit
Take profit: Reduce to half at 2R, trail at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_volatility_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, adjust=False, min_periods=half).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


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
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel"""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest


def calculate_atr_percentile(atr, window=100):
    """Calculate rolling percentile rank of ATR"""
    n = len(atr)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(atr[i]):
            window_data = atr[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= atr[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_high, donchian_low = calculate_donchian(high, low, 20)
    
    # ATR percentile for volatility filter
    atr_pr = calculate_atr_percentile(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% of capital
    MAX_SIZE = 0.30   # 30% max on strong signals
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 120  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_pr[i]) or 
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter (HTF)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volatility filter: only trade when ATR is in 20th-80th percentile
        # Avoid extreme volatility (bottom 20% = dead market, top 20% = chaos)
        volatility_ok = 0.20 <= atr_pr[i] <= 0.80
        
        # Donchian breakout confirmation
        donchian_breakout_long = close[i] > donchian_high[i - 1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_low[i - 1] if i > 0 else False
        
        # RSI pullback zone (45-55 for entries within trend)
        rsi_neutral = 45 <= rsi[i] <= 55
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Determine target signal
        target_signal = 0.0
        signal_strength = BASE_SIZE
        
        # Long entry: 4h trend up + volatility ok + RSI bullish + Donchian breakout OR RSI pullback
        if hma_trend == 1 and volatility_ok:
            if donchian_breakout_long and rsi_bullish:
                signal_strength = MAX_SIZE  # Strong breakout
            elif rsi_neutral and rsi_bullish:
                signal_strength = BASE_SIZE  # Pullback entry
            else:
                signal_strength = 0.0
            
            if signal_strength > 0:
                target_signal = signal_strength
        
        # Short entry: 4h trend down + volatility ok + RSI bearish + Donchian breakout OR RSI pullback
        elif hma_trend == -1 and volatility_ok:
            if donchian_breakout_short and rsi_bearish:
                signal_strength = MAX_SIZE  # Strong breakout
            elif rsi_neutral and rsi_bearish:
                signal_strength = BASE_SIZE  # Pullback entry
            else:
                signal_strength = 0.0
            
            if signal_strength > 0:
                target_signal = -signal_strength
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            r = 2.0 * entry_atr  # Risk distance
            
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 2.0 * r:  # 2R = 4*ATR
                        take_profit_triggered = True
                
                # Check trend reversal
                if hma_trend == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 2.0 * r:  # 2R profit
                        take_profit_triggered = True
                
                # Check trend reversal
                if hma_trend == 1:
                    trend_reversal = True
        
        # Apply signals based on conditions
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
            
        elif trend_reversal:
            # HTF trend reversed, exit position
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        else:
            # Normal signal application
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
                # Maintain or adjust existing position
                if target_signal == 0.0:
                    # No signal, maintain position
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
                elif np.sign(target_signal) == position_side:
                    # Same direction, maintain or increase
                    signals[i] = target_signal
                else:
                    # Opposite direction, exit
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
            else:
                signals[i] = 0.0
    
    return signals