#!/usr/bin/env python3
"""
EXPERIMENT #007 - RSI Mean Reversion + Dual HTF Trend Filter (15m primary)
=====================================================================================
Hypothesis: 15m RSI extremes (oversold/overbought) capture pullbacks in strong trends.
Unlike breakout strategies (Donchian/Supertrend) which failed with low trade counts,
mean reversion within trends generates more signals on 15m. Dual HTF filter (1h + 4h HMA)
ensures we only trade pullbacks in the direction of the major trend.

Key features:
- Primary TF: 15m (required for this experiment)
- HTF filters: 1h HMA(21) + 4h HMA(21) for trend alignment
- Entry: RSI(14) < 35 (long) or > 65 (short) with HTF confirmation
- Regime: Both 1h and 4h must align (same trend direction)
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2R profit, trail stop at 1R
- Volume filter: Entry volume > 0.8 * 20-period avg (confirms interest)

Why this should beat previous failures:
- Mean reversion generates MORE trades than breakouts on 15m
- RSI extremes are proven entry signals (not novel/untested)
- Dual HTF (1h+4h) is less restrictive than triple HTF (more trades)
- Conservative sizing (0.25-0.30) controls drawdown better than 1.0
- Volume filter reduces false signals in low-liquidity periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "rsi_meanrev_dualhtf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (EMA with span=period)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
            rsi[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
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


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_sma = calculate_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
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
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]) or
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend alignment (both 1h and 4h must agree)
        price_above_1h_hma = close[i] > hma_1h_aligned[i]
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        
        # HTF trend direction
        htf_trend = 0
        if price_above_1h_hma and price_above_4h_hma:
            htf_trend = 1  # Bullish alignment
        elif not price_above_1h_hma and not price_above_4h_hma:
            htf_trend = -1  # Bearish alignment
        # else: htf_trend = 0 (mixed, no trade)
        
        # RSI signals (mean reversion within trend)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Volume confirmation (entry volume > 80% of 20-period avg)
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # ATR-based position sizing adjustment
        atr_pct = atr[i] / close[i] * 100
        # Reduce size if ATR is very high (>3% of price)
        if atr_pct > 3.0:
            position_size = BASE_SIZE * 0.8
        else:
            position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: RSI oversold + HTF bullish + volume confirmed
        if rsi_oversold and htf_trend == 1 and volume_confirmed:
            target_signal = position_size
        
        # Short entry: RSI overbought + HTF bearish + volume confirmed
        elif rsi_overbought and htf_trend == -1 and volume_confirmed:
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
                # Exit if HTF alignment breaks OR RSI crosses back through 50
                htf_alignment_broken = (position_side == 1 and htf_trend == -1) or \
                                       (position_side == -1 and htf_trend == 1)
                rsi_reversal = (position_side == 1 and rsi[i] > 55) or \
                               (position_side == -1 and rsi[i] < 45)
                
                if htf_alignment_broken or rsi_reversal:
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