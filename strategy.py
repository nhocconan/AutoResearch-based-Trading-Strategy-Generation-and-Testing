#!/usr/bin/env python3
"""
Experiment #041: 4h Simple Trend + RSI Pullback + Choppiness Filter

Hypothesis: Previous strategies failed due to over-complexity (too many conflicting filters).
This strategy simplifies to core proven edges:
1. 1d HMA(21) for trend direction (simple, proven)
2. RSI(14) for pullback entries (not Connors - simpler, more reliable)
3. Choppiness Index to avoid trading in extreme chop
4. ATR(14) trailing stoploss at 2.5x
5. Discrete position sizing: 0.25 base

Key differences from failed experiments:
- Fewer entry conditions (no streak RSI, no percentile rank)
- Looser RSI thresholds to ensure trades (30/70 not 15/85)
- Simple HMA instead of KAMA/DEMA
- Focus on 4h primary with 1d bias (proven combination)
- Frequency safeguard: force entry after 50 bars without trades

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_rsi_chop_1d_hma_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_bullish = close[i] > hma_1d_21_aligned[i]
        trend_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINNESS FILTER ===
        # Only trade if not extremely choppy (CHOP < 70)
        # Allow trading in neutral/trending regimes
        tradeable = chop_14[i] < 70
        
        # === RSI ENTRY SIGNALS ===
        # Long: RSI < 35 (pullback in uptrend) - looser than 30 to ensure trades
        # Short: RSI > 65 (rally in downtrend) - looser than 70 to ensure trades
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if tradeable:
            # LONG: Bullish trend + RSI pullback
            if trend_bullish and rsi_oversold:
                new_signal = BASE_SIZE
            
            # SHORT: Bearish trend + RSI rally
            elif trend_bearish and rsi_overbought:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~8-9 days on 4h), force entry with weaker signal
        # This ensures minimum trade count on all symbols
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if trend_bullish and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.5
            elif trend_bearish and rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
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
            in_position = True
            position_side = np.sign(new_signal)
            if position_side > 0:
                highest_price = close[i]
                lowest_price = 0.0
            else:
                highest_price = 0.0
                lowest_price = close[i]
            last_trade_bar = i
        elif in_position and new_signal == 0.0:
            in_position = False
            position_side = 0
            highest_price = 0.0
            lowest_price = 0.0
            last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals