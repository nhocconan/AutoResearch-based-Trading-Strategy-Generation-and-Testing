#!/usr/bin/env python3
"""
EXPERIMENT #087 - KAMA Adaptive Trend + RSI Pullback + 4h HMA Filter (1h primary)
==================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to crypto volatility better
than fixed-period MAs. On 1h timeframe, KAMA crossovers with RSI pullback entries
filtered by 4h HMA trend and Bollinger Band regime should capture trends while
avoiding chop. KAMA's Efficiency Ratio reduces whipsaws in sideways markets.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(10) crossover with EMA(21) confirmation
- Entry: RSI(14) pullback to 40-60 zone in direction of trend
- Regime: Bollinger Band Width percentile > 40th (avoid extreme squeeze/expansion)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 base, 0.30 max with strong signals (discrete levels)
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- KAMA adapts smoothing based on market noise (ER)
- 1h timeframe captures more opportunities than 12h while avoiding 15m noise
- RSI pullback entries reduce chasing breakouts
- 4h HMA filter ensures we trade with higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_4hhtf_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market Efficiency Ratio (ER)
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    Higher ER = trending market = faster smoothing
    Lower ER = choppy market = slower smoothing
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        signal = abs(close[i] - close[i - period + 1])
        noise = 0.0
        for j in range(i - period + 2, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    # SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.zeros(n)
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
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
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
            loss[i] = 0.0
        else:
            gain[i] = 0.0
            loss[i] = abs(delta[i - 1])
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    bandwidth[:] = np.nan
    
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    for i in range(period - 1, n):
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
    
    return upper, lower, bandwidth


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    ema_21 = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BB Width percentile rank (regime filter)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong signals
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or np.isnan(ema_21[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA vs EMA crossover (trend confirmation)
        kama_above_ema = kama[i] > ema_21[i]
        
        # RSI pullback zone (40-60 for continuation, <40 for long entry, >60 for short)
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # BB Width regime (avoid extreme squeeze or expansion)
        bb_regime_ok = bb_width_pr[i] > 0.30 and bb_width_pr[i] < 0.85
        
        # Price position relative to Bollinger Bands
        price_vs_bb_long = close[i] > bb_lower[i] and close[i] < bb_upper[i]
        price_vs_bb_short = close[i] > bb_lower[i] and close[i] < bb_upper[i]
        
        # Calculate position size based on signal strength
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA > EMA + 4h HMA bullish + RSI pullback or oversold + BB regime OK
        if (kama_above_ema and hma_trend == 1 and bb_regime_ok and
            (rsi_oversold or (rsi_neutral and rsi[i] > 45))):
            # Stronger signal if RSI oversold
            if rsi_oversold:
                position_size = MAX_SIZE
            target_signal = position_size
        
        # Short entry: KAMA < EMA + 4h HMA bearish + RSI pullback or overbought + BB regime OK
        elif (not kama_above_ema and hma_trend == -1 and bb_regime_ok and
              (rsi_overbought or (rsi_neutral and rsi[i] < 55))):
            # Stronger signal if RSI overbought
            if rsi_overbought:
                position_size = MAX_SIZE
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
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
                # Exit if KAMA/EMA crossover reverses OR 4h HMA alignment breaks
                kama_ema_reversal_long = not kama_above_ema
                kama_ema_reversal_short = kama_above_ema
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if kama_ema_reversal_long or kama_ema_reversal_short or hma_alignment_broken:
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