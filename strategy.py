#!/usr/bin/env python3
"""
EXPERIMENT #074 - KAMA Adaptive Trend + RSI Pullback + 4h HTF Filter (30m primary)
=====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than 
EMA/HMA, reducing whipsaws in chop. RSI pullback entries (not breakouts) work better 
on 30m timeframe - buy dips in uptrend, sell rallies in downtrend. 4h HMA provides 
major trend filter. Volume confirmation (taker_buy_ratio) adds conviction.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(50) for major trend direction
- Trend: KAMA(10,2,30) adaptive moving average
- Entry: RSI(14) pullback to 40-50 zone in uptrend (or 50-60 in downtrend)
- Volume: taker_buy_volume ratio > 0.55 for longs, < 0.45 for shorts
- Regime: KAMA slope + price position relative to KAMA
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to volatility, fewer false signals in chop
- RSI pullback entries have better risk/reward than breakouts on 30m
- 4h HTF filter ensures we trade with major trend
- Volume confirmation reduces false entries
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_vol_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - smooth in trends, flat in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with first close after er_period
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.diff(close)
    
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
        else:
            loss[i] = -delta[i - 1]
    
    # Wilder's smoothing (EMA with span=period)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (rate of change)"""
    n = len(kama)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if kama[i - lookback] != 0:
            slope[i] = (kama[i] - kama[i - lookback]) / kama[i - lookback] * 100
        else:
            slope[i] = 0
    
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_vol = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    kama_slope = calculate_kama_slope(kama, lookback=5)
    
    # Calculate taker buy ratio
    taker_buy_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_buy_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            taker_buy_ratio[i] = 0.5
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(kama_slope[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction
        price_above_kama = close[i] > kama[i]
        kama_uptrend = kama_slope[i] > 0.0
        kama_downtrend = kama_slope[i] < 0.0
        
        # RSI pullback zones
        rsi_pullback_long = 40 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # Rally in downtrend
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Volume confirmation
        volume_bullish = taker_buy_ratio[i] > 0.55
        volume_bearish = taker_buy_ratio[i] < 0.45
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend up + KAMA uptrend + RSI pullback + volume bullish
        if (hma_trend == 1 and price_above_kama and kama_uptrend and 
            rsi_pullback_long and volume_bullish):
            target_signal = position_size
        
        # Strong long: RSI oversold + all other conditions
        elif (hma_trend == 1 and price_above_kama and kama_uptrend and 
              rsi_oversold and volume_bullish):
            target_signal = MAX_SIZE
        
        # Short entry: 4h trend down + KAMA downtrend + RSI pullback + volume bearish
        elif (hma_trend == -1 and not price_above_kama and kama_downtrend and 
              rsi_pullback_short and volume_bearish):
            target_signal = -position_size
        
        # Strong short: RSI overbought + all other conditions
        elif (hma_trend == -1 and not price_above_kama and kama_downtrend and 
              rsi_overbought and volume_bearish):
            target_signal = -MAX_SIZE
        
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
                # Exit if KAMA trend reverses OR 4h HTF alignment breaks
                kama_reversal_long = (position_side == 1 and kama_downtrend and close[i] < kama[i])
                kama_reversal_short = (position_side == -1 and kama_uptrend and close[i] > kama[i])
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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