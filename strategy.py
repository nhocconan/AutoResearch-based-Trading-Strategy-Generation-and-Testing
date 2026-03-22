#!/usr/bin/env python3
"""
Experiment #025: 15m 4h HMA Trend + 1h RSI Pullback + Choppiness Regime Filter

Hypothesis: After analyzing 24 experiments, the pattern is clear:
1. Lower TFs (15m-1h) fail due to noise and fee drag WITHOUT strong HTF filter
2. 4h HMA trend filter has proven robust across multiple experiments
3. RSI pullback entries within HTF trend catch better risk/reward entries
4. Choppiness Index (CHOP) regime filter is UNDERUTILIZED - only 2 attempts
5. CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
6. This regime awareness should avoid whipsaws that destroyed previous 15m strategies

Strategy combines:
1. 4h HMA(21): Primary trend bias - long only when price > 4h HMA, short when <
2. 1h RSI(14) pullback: Enter on RSI 35-45 in uptrend, 55-65 in downtrend
3. Choppiness Index(14): Only trade when CHOP < 55 (trending regime)
4. Volume confirmation: Volume > 1.2x 20-bar average
5. ATR(14) stoploss: 2.5x ATR trailing stop (tighter for 15m TF)
6. Discrete sizing: 0.30 strong signal, 0.20 moderate

Why this should beat current best (Sharpe=0.137):
- 4h HMA provides STRONGER trend filter than 1h (used in failed #013, #019)
- RSI pullback entries have better R:R than breakouts (less chasing)
- CHOP regime filter addresses the #1 failure mode (trading in chop)
- Conservative 15m position sizing (0.20-0.30) limits drawdown
- Target 50-80 trades/year on 15m (optimal per Rule 10)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h and 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_chop_regime_vol_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_volume_spike(volume, lookback=20, threshold=1.2):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.2)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.30  # All filters agree
    SIZE_MODERATE = 0.20  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_vs_4h = close[i] - hma_4h_aligned[i]
        bull_4h = price_vs_4h > 0
        bear_4h = price_vs_4h < 0
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 55 = trending regime (trade with trend)
        # CHOP > 55 = ranging regime (reduce position or skip)
        trending_regime = chop_14[i] < 55
        strong_trend = chop_14[i] < 45
        
        # === 1h RSI PULLBACK ENTRY ===
        # In uptrend: enter on RSI pullback to 35-50
        # In downtrend: enter on RSI bounce to 50-65
        rsi_1h_val = rsi_1h_aligned[i]
        rsi_pullback_long = 35 <= rsi_1h_val <= 50
        rsi_pullback_short = 50 <= rsi_1h_val <= 65
        
        # === 15m RSI CONFIRMATION ===
        rsi_15m_val = rsi_14[i]
        rsi_15m_long = rsi_15m_val < 55  # Not overbought
        rsi_15m_short = rsi_15m_val > 45  # Not oversold
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY: 4h bull + trending regime + 1h RSI pullback + 15m RSI confirm
        if bull_4h and trending_regime and rsi_pullback_long:
            signal_strength += 2  # 4h trend + regime filter (core signals)
            
            if rsi_15m_long:
                signal_strength += 1  # 15m RSI confirmation
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if strong_trend:
                signal_strength += 1  # Strong trending regime
            
            # Assign size based on confirmation count
            if signal_strength >= 4:
                new_signal = SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: 4h bear + trending regime + 1h RSI pullback + 15m RSI confirm
        elif bear_4h and trending_regime and rsi_pullback_short:
            signal_strength += 2  # 4h trend + regime filter (core signals)
            
            if rsi_15m_short:
                signal_strength += 1  # 15m RSI confirmation
            
            if vol_confirmed:
                signal_strength += 1  # Volume confirmation
            
            if strong_trend:
                signal_strength += 1  # Strong trending regime
            
            # Assign size based on confirmation count
            if signal_strength >= 4:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME EXIT ===
        # Exit if regime turns choppy (CHOP > 65)
        regime_exit = False
        if in_position and chop_14[i] > 65:
            regime_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if 4h trend strongly reverses against position
            if position_side > 0 and bear_4h:
                trend_exit = True
            if position_side < 0 and bull_4h:
                trend_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals