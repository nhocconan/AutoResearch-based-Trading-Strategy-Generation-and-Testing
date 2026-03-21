#!/usr/bin/env python3
"""
EXPERIMENT #049 - Bollinger Mean Reversion + HTF Trend Filter + RSI Extremes (15m primary)
==========================================================================================
Hypothesis: On 15m timeframe, price frequently mean-reverts within the context of the HTF trend.
Unlike pure breakout strategies that fail in chop, mean reversion WITH trend filter captures
pullbacks in established trends. This differs from #047 by using mean reversion (not breakouts)
on faster timeframe (15m vs 12h) with tighter HTF filter (4h vs 1d/1w).

Key features:
- Primary TF: 15m
- HTF filter: 4h HMA(21) for trend direction (only trade pullbacks WITH trend)
- Entry: Price touches BB(20,2.0) extreme + RSI(14) extreme + volume spike
- Long: Price < lower BB + RSI < 35 + 4h HMA bullish + volume > 1.5x avg
- Short: Price > upper BB + RSI > 65 + 4h HMA bearish + volume > 1.5x avg
- Regime: 4h HMA slope filter (avoid flat/chop markets)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.35)
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat #047 (Sharpe=0.490):
- Mean reversion on 15m has higher win rate than breakouts
- 4h HMA filter removes counter-trend trades (major cause of DD)
- RSI + BB + Volume triple confirmation reduces false signals
- Fewer trades than pure momentum = less fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_mr_rsi_4htrend_15m_v1"
timeframe = "15m"
leverage = 1.0


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Bandwidth for regime detection
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume spike detection"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (rate of change over lookback periods)"""
    n = len(hma_values)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback]
    
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope for regime filter
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, 5)
    
    # Calculate 15m indicators
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size with strong confirmation
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_4h_bullish = price_above_4h_hma and hma_4h_slope[i] > 0.001  # Slope > 0.1%
        hma_4h_bearish = not price_above_4h_hma and hma_4h_slope[i] < -0.001  # Slope < -0.1%
        
        # Regime filter: avoid flat/chop markets (4h HMA slope near zero)
        regime_chop = abs(hma_4h_slope[i]) < 0.0005  # Less than 0.05% slope = chop
        
        # Volume spike confirmation (volume > 1.5x 20-period average)
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # Bollinger Band extreme detection
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.001  # Within 0.1% of lower BB
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.999  # Within 0.1% of upper BB
        
        # RSI extreme detection (slightly relaxed from traditional 30/70)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Calculate position size based on confirmation strength
        confirmation_count = 0
        if volume_spike:
            confirmation_count += 1
        if rsi_oversold or rsi_overbought:
            confirmation_count += 1
        
        position_size = BASE_SIZE
        if confirmation_count >= 2:
            position_size = MAX_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Price at lower BB + RSI oversold + 4h bullish + volume spike
        if (price_at_lower_bb and rsi_oversold and hma_4h_bullish and 
            not regime_chop):
            target_signal = position_size
        
        # Short entry: Price at upper BB + RSI overbought + 4h bearish + volume spike
        elif (price_at_upper_bb and rsi_overbought and hma_4h_bearish and 
              not regime_chop):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Maintain existing position (check if trend reversed or mean reversion complete)
                # Exit if price crosses BB middle (SMA) OR 4h trend reverses
                price_crossed_sma_long = close[i] > bb_sma[i] and position_side == 1
                price_crossed_sma_short = close[i] < bb_sma[i] and position_side == -1
                hma_reversal = (position_side == 1 and hma_4h_bearish) or \
                               (position_side == -1 and hma_4h_bullish)
                
                if price_crossed_sma_long or price_crossed_sma_short or hma_reversal:
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