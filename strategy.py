#!/usr/bin/env python3
"""
Experiment #015: 1h Supertrend + 4h HMA Trend + RSI Pullback + ADX Filter

Hypothesis: After analyzing 14 failed experiments, the pattern shows:
1. Mean reversion strategies (CRSI, Fisher) fail in crypto's persistent trends
2. Lower TFs (15m/30m) suffer from noise and fee drag
3. The current best (30m Supertrend + 4h HMA + RSI) has Sharpe=0.123

This 1h strategy simplifies and improves:

1. 4h HMA(21) trend bias: Stable HTF filter. Only long if price>4h_HMA,
   only short if price<4h_HMA. Proven in best performing strategy.

2. 1h Supertrend(10,3): Primary entry signal. Catches trend continuations
   with less whipsaw than EMA crossover. Standard parameters from literature.

3. 1h RSI(14) pullback: Enter on dips in uptrend (RSI 40-55) or rallies
   in downtrend (RSI 45-60). Avoids chasing tops/bottoms.

4. 1h ADX(14) filter: ADX>20 confirms trend strength. Avoids choppy markets
   where Supertrend whipsaws. Lower threshold than failed strategies (was 25-30).

5. ATR(14) trailing stop: 2.5*ATR stoploss to protect from crashes.
   Critical after 2022 -77% drawdown lesson.

6. Conservative sizing: 0.25 base, 0.30 max. Discrete levels to minimize
   fee churn. Each signal change costs 0.05% round trip.

Why this should beat current best (Sharpe=0.123):
- 1h TF balances signal quality vs fee drag (30-60 trades/year target)
- Simpler logic = more consistent triggers (avoid 0-trade failure)
- Supertrend proven in best strategy, now with better RSI pullback timing
- ADX threshold lowered to 20 (was 25+) to generate more signals
- 4h HMA more stable than 12h/1d for 1h primary timeframe

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year (within 1h optimal range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_rsi_pullback_adx_atr_v1"
timeframe = "1h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values using Wilder's method
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=below price=bullish, -1=above price=bearish)
    """
    atr = calculate_atr(high, low, close, period)
    
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    # Initial values
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    
    for i in range(1, n):
        # Calculate final upper/lower
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine direction and supertrend value
        if direction[i-1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    MAX_SIZE = 0.30
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            continue
        
        # === 4H HMA TREND BIAS (HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === ADX TREND STRENGTH FILTER ===
        # ADX > 20 = trending market (lower threshold for more signals)
        is_trending = adx_14[i] > 20
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1  # Price above supertrend
        st_bearish = st_direction[i] == -1  # Price below supertrend
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI 40-55 (pullback in uptrend, not oversold)
        # Short: RSI 45-60 (rally in downtrend, not overbought)
        rsi_pullback_long = 40 <= rsi_14[i] <= 55
        rsi_pullback_short = 45 <= rsi_14[i] <= 60
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 4H bull bias + Supertrend bullish + RSI pullback + ADX trending
        if bull_bias and st_bullish and rsi_pullback_long and is_trending:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: 4H bear bias + Supertrend bearish + RSI pullback + ADX trending
        elif bear_bias and st_bearish and rsi_pullback_short and is_trending:
            new_signal = -BASE_SIZE
        
        # STRONG SIGNAL: All conditions aligned + strong ADX
        if is_trending and adx_14[i] > 30:
            if bull_bias and st_bullish:
                new_signal = max(new_signal, MAX_SIZE)
            elif bear_bias and st_bearish:
                new_signal = min(new_signal, -MAX_SIZE)
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if Supertrend flips bearish
            if position_side > 0 and st_bearish:
                trend_reversal = True
            # Exit short if Supertrend flips bullish
            if position_side < 0 and st_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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