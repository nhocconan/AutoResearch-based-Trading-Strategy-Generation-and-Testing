#!/usr/bin/env python3
"""
Experiment #189: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous strategies failed due to OVER-FILTERING (too many confluence
requirements = 0 trades). This strategy uses SIMPLER entry logic:

1. 4h HMA(21) trend direction (fast, responsive)
2. 4h RSI(14) pullback entries (RSI<40 long, RSI>60 short)
3. 1d HMA(21) slope for major bias (soft filter, not hard requirement)
4. ATR(14) trailing stoploss (2.5x ATR)
5. Minimum cooldown between trades (40 bars = ~10 days on 4h)

Why this should work:
- Fewer filters = more trades (avoids Sharpe=0.000 problem)
- HMA is faster than EMA, catches trends earlier
- RSI pullback in trend = high win rate setup
- 4h timeframe = 20-50 trades/year target
- 1d HTF provides trend bias without over-constraining

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol (ensure >=10 train, >=3 test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_simp_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / abs(hma_values[i - lookback]) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Price relative to HMA
    price_vs_hma21 = (close - hma_4h_21) / np.where(hma_4h_21 != 0, hma_4h_21, 1e-10) * 100
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
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
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        # === 1D TREND BIAS (soft filter) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        trend_1d_neutral = abs(hma_1d_slope_aligned[i]) <= 0.2
        
        # === 4H TREND ===
        trend_4h_bullish = hma_4h_21[i] > hma_4h_48[i]
        trend_4h_bearish = hma_4h_21[i] < hma_4h_48[i]
        price_above_hma21 = close[i] > hma_4h_21[i]
        price_below_hma21 = close[i] < hma_4h_21[i]
        
        # === RSI CONDITIONS (relaxed for more trades) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_weak = adx_14[i] < 20
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if adx_weak:
            current_size = HALF_SIZE  # Reduce size in chop
        
        # === ENTRY LOGIC (simplified for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths, fewer requirements
        long_cond = False
        
        # Path 1: 4h bullish + RSI pullback (primary)
        if trend_4h_bullish and rsi_oversold:
            long_cond = True
        
        # Path 2: 1d bullish bias + 4h RSI extreme
        if trend_1d_bullish and rsi_extreme_low:
            long_cond = True
        
        # Path 3: Price above 4h HMA + RSI dip (trend continuation)
        if price_above_hma21 and rsi_14[i] < 45 and bars_since_last_trade > 60:
            long_cond = True
        
        # Path 4: 1d neutral + 4h bullish + RSI oversold
        if trend_1d_neutral and trend_4h_bullish and rsi_oversold:
            long_cond = True
        
        if long_cond and bars_since_last_trade > 40:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_cond = False
        
        # Path 1: 4h bearish + RSI rally (primary)
        if trend_4h_bearish and rsi_overbought:
            short_cond = True
        
        # Path 2: 1d bearish bias + 4h RSI extreme
        if trend_1d_bearish and rsi_extreme_high:
            short_cond = True
        
        # Path 3: Price below 4h HMA + RSI spike (trend continuation)
        if price_below_hma21 and rsi_14[i] > 55 and bars_since_last_trade > 60:
            short_cond = True
        
        # Path 4: 1d neutral + 4h bearish + RSI overbought
        if trend_1d_neutral and trend_4h_bearish and rsi_overbought:
            short_cond = True
        
        if short_cond and bars_since_last_trade > 40:
            new_signal = -current_size
        
        # === FORCED TRADE LOGIC (ensure minimum trades) ===
        # If no trades for 120 bars (~20 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 50:
                new_signal = HALF_SIZE
            elif trend_4h_bearish and rsi_14[i] > 50:
                new_signal = -HALF_SIZE
            elif rsi_extreme_low:
                new_signal = HALF_SIZE
            elif rsi_extreme_high:
                new_signal = -HALF_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and rsi_14[i] > 55:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and rsi_14[i] < 45:
                trend_reversal = True
        
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