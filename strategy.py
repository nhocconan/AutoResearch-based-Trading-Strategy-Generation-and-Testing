#!/usr/bin/env python3
"""
Experiment #101: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Mean Reversion

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than fixed EMA/HMA. In trending markets it follows price closely, in choppy markets
it flattens to avoid whipsaws. Combined with RSI extremes and 1d trend bias, this
should generate consistent trades with good win rate.

Key improvements over failed strategies:
1. LOOSER entry thresholds (RSI <40/>60 instead of <30/>70) to ensure trades generate
2. Volume filter is permissive (>0.8x avg, not >1.5x)
3. Frequency safeguard: force entry after 80 bars without trade
4. Maintain position signal across bars (don't flip to 0 each bar)
5. Simple 1d HMA bias (price above/below) instead of complex slope calculation

Strategy Logic:
1. KAMA(14/50) on 4h: Adaptive trend following
2. RSI(14): <40 oversold, >60 overbought (loose thresholds)
3. 1d HMA(21): Major trend bias (price above = long bias, below = short bias)
4. Volume filter: Current volume > 0.8 * 20-bar avg (permissive)
5. ATR(14) 2.5x trailing stop
6. Position size: 0.30 discrete

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 30-60/year per symbol (looser entries than 12h strategies)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_hma1d_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - follows closely in trends, flattens in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(period))
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    volatility = volatility.replace(0, np.nan)
    er = change / volatility
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    kama_14 = calculate_kama(close, 14, 2, 30)
    kama_50 = calculate_kama(close, 50, 2, 30)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(kama_50[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND (KAMA) ===
        kama_bullish = kama_14[i] > kama_50[i]
        kama_bearish = kama_14[i] < kama_50[i]
        
        # === RSI SIGNALS (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === VOLUME FILTER (PERMISSIVE) ===
        vol_confirm = volume[i] > 0.8 * vol_avg_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - 1d bullish + 4h KAMA bullish + RSI oversold
        if price_above_1d_hma and kama_bullish and rsi_oversold:
            new_signal = BASE_SIZE
        # Alternative long: 1d bullish + RSI very oversold (no KAMA filter)
        elif price_above_1d_hma and rsi_14[i] < 30:
            new_signal = BASE_SIZE
        # Alternative long: KAMA bullish crossover + RSI moderate
        elif kama_bullish and rsi_14[i] < 45 and vol_confirm:
            new_signal = BASE_SIZE * 0.8
        
        # SHORT ENTRIES - 1d bearish + 4h KAMA bearish + RSI overbought
        if price_below_1d_hma and kama_bearish and rsi_overbought:
            new_signal = -BASE_SIZE
        # Alternative short: 1d bearish + RSI very overbought (no KAMA filter)
        elif price_below_1d_hma and rsi_14[i] > 70:
            new_signal = -BASE_SIZE
        # Alternative short: KAMA bearish crossover + RSI moderate
        elif kama_bearish and rsi_14[i] > 55 and vol_confirm:
            new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~2 weeks on 4h), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if price_above_1d_hma and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.5
            elif price_below_1d_hma and rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.5
        
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
        
        # Apply stoploss
        if stoploss_triggered:
            new_signal = 0.0
        
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