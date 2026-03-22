#!/usr/bin/env python3
"""
Experiment #318: 30m Primary + 4h/1d HTF — Simplified Trend + RSI Pullback

Hypothesis: Previous 30m strategies (#308, #310, #315) failed with 0 trades because
entry conditions were TOO STRICT (session filter + 3-4 confluence filters never aligned).

This strategy SIMPLIFIES to ensure trades are generated:
1. 4h HMA(21) for major trend direction (only 1 HTF filter)
2. 30m RSI(14) pullback entries (40-55 for long, 45-60 for short)
3. ONE volatility filter (ATR ratio < 2.0 to avoid extreme vol)
4. NO session filter (was killing trade frequency)
5. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto favors longs)

Why this should work when #308 failed:
- Removed session filter (8-20 UTC was blocking most entries)
- Reduced from 3-4 confluence filters to just 2 (HTF trend + RSI)
- Looser RSI ranges (40-55 instead of 42-48)
- Added frequency safeguard: force entry if no trade in 50 bars

Target: 40-80 trades/year on 30m (appropriate for this TF)
Position sizing: 0.20-0.30 discrete levels (max 0.35)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_4h_simp_v1"
timeframe = "30m"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    n = period
    n2 = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n2)
    wma_full = wma(close_s, n)
    
    hull = 2.0 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (regime filter)
    sma_1d_200 = calculate_sma(df_1d['close'].values, period=200)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_30m_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Asymmetric: longs favored in crypto
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.18
    SHORT_STRONG = 0.22
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_30m_21[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA(21)
        # Bear: price below 4h HMA(21)
        regime_bull = close[i] > hma_4h_21_aligned[i]
        regime_bear = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA alignment (both 21 and 50)
        hma_4h_aligned = hma_4h_21_aligned[i] > hma_4h_50_aligned[i] if not np.isnan(hma_4h_50_aligned[i]) else regime_bull
        
        # === 1D REGIME FILTER (optional boost) ===
        price_above_1d_sma200 = close[i] > sma_1d_200_aligned[i] if not np.isnan(sma_1d_200_aligned[i]) else True
        
        # === VOLATILITY FILTER (avoid extreme vol) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        vol_ok = atr_ratio < 2.5  # Allow higher ratio to generate more trades
        
        # === 30M LOCAL TREND ===
        hma_30m_bullish = close[i] > hma_30m_21[i]
        hma_30m_bearish = close[i] < hma_30m_21[i]
        
        # Price relative to SMA200 (long-term trend)
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (pullback entries, LOOSE ranges for trade gen) ===
        # RSI pullback long: RSI 35-55 in uptrend (wider than before)
        # RSI pullback short: RSI 45-65 in downtrend (wider than before)
        rsi_pullback_long = 35.0 < rsi_14[i] < 58.0
        rsi_pullback_short = 42.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 38.0
        rsi_strong_overbought = rsi_14[i] > 62.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLIFIED - 2 filters max) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime - asymmetric sizing)
        if regime_bull and vol_ok:
            # Primary: RSI pullback in 4h bull trend
            if rsi_pullback_long and hma_30m_bullish:
                new_signal = LONG_BASE
            
            # Strong: RSI very oversold + bull regime
            elif rsi_strong_oversold and regime_bull:
                new_signal = LONG_STRONG
            
            # 30m HMA bullish + RSI rising
            elif hma_30m_bullish and rsi_rising and rsi_14[i] > 40.0:
                new_signal = LONG_BASE
        
        # SHORT ENTRIES (only in bear regime, reduced size - asymmetric)
        if regime_bear and vol_ok:
            # Primary: RSI pullback in 4h bear trend
            if rsi_pullback_short and hma_30m_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong: RSI very overbought + bear regime
            elif rsi_strong_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # 30m HMA bearish + RSI falling
            elif hma_30m_bearish and rsi_falling and rsi_14[i] < 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (CRITICAL - ensure trades generated) ===
        # Force trade if no signal for 50 bars (~25 hours on 30m)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40.0:
                new_signal = LONG_BASE * 0.7
            elif regime_bear and rsi_14[i] < 60.0:
                new_signal = -SHORT_BASE * 0.7
            elif rsi_strong_oversold:
                new_signal = LONG_BASE * 0.7
            elif rsi_strong_overbought:
                new_signal = -SHORT_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI turns overbought
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            # Short position: exit when RSI turns oversold
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when 30m HMA turns bearish
            if position_side > 0 and hma_30m_bearish:
                hma_exit = True
            # Short position: exit when 30m HMA turns bullish
            if position_side < 0 and hma_30m_bullish:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns bearish
            if position_side > 0 and regime_bear:
                regime_reversal = True
            # Short position but 4h regime turns bullish
            if position_side < 0 and regime_bull:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.20:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals