#!/usr/bin/env python3
"""
Experiment #031: 4h Primary + 1d/1w HTF — Donchian Breakout with HMA Trend

Hypothesis: Previous failures due to over-filtering (too many conditions = 0 trades).
This strategy uses SIMPLER confluence for reliable trade generation:

1. 1d HMA(21) for MAJOR trend bias (only trade WITH 1d trend)
2. 1w HMA(21) for SECULAR trend confirmation (avoid counter-trend in bear)
3. Donchian(20) breakout for entry timing (proven on SOL)
4. RSI(14) momentum filter (40-60 neutral, >55 long, <45 short)
5. ATR(14) trailing stoploss at 2.5x
6. Position sizing: 0.25-0.30 discrete based on trend strength

Why this should work:
- Donchian breakouts catch sustained moves (not whipsaws)
- 1d + 1w HMA alignment filters out counter-trend trades
- RSI filter avoids entering at exhaustion
- 4h timeframe = 20-50 trades/year target (fee manageable)
- Simpler logic = more trades generated (avoid 0-trade failure)

Timeframe: 4h (REQUIRED per experiment #031)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # 4h HMA for intermediate trend
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W SECULAR TREND (MAJOR BIAS) ===
        # Price above 1w HMA = bull market (prefer longs)
        # Price below 1w HMA = bear market (prefer shorts)
        secular_bull = close[i] > hma_1w_21_aligned[i]
        secular_bear = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H SHORT-TERM TREND ===
        trend_4h_bullish = close[i] > hma_4h_21[i]
        trend_4h_bearish = close[i] < hma_4h_21[i]
        
        # === RSI MOMENTUM FILTER ===
        # RSI > 55 = bullish momentum
        # RSI < 45 = bearish momentum
        # 45-55 = neutral (no entry)
        rsi_bullish = rsi_14[i] > 55
        rsi_bearish = rsi_14[i] < 45
        
        # === DONCHIAN BREAKOUT ===
        # Break above upper = long signal
        # Break below lower = short signal
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING BASED ON TREND CONFLUENCE ===
        # Strong signal: all 3 TF aligned (1w + 1d + 4h)
        # Normal signal: 1d + 4h aligned
        long_confluence_strong = secular_bull and trend_1d_bullish and trend_4h_bullish
        short_confluence_strong = secular_bear and trend_1d_bearish and trend_4h_bearish
        
        long_confluence_normal = trend_1d_bullish and trend_4h_bullish
        short_confluence_normal = trend_1d_bearish and trend_4h_bearish
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: Donchian breakout + RSI momentum + trend alignment
        if breakout_long and rsi_bullish:
            if long_confluence_strong:
                new_signal = STRONG_SIZE
            elif long_confluence_normal:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Require: Donchian breakout + RSI momentum + trend alignment
        if breakout_short and rsi_bearish:
            if short_confluence_strong:
                new_signal = -STRONG_SIZE
            elif short_confluence_normal:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        # This ensures we generate minimum trades (Rule 9)
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and rsi_14[i] > 50:
                new_signal = BASE_SIZE * 0.8
            elif trend_1d_bearish and trend_4h_bearish and rsi_14[i] < 50:
                new_signal = -BASE_SIZE * 0.8
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if major trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] < 40:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] > 60:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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
                # Flip position
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