#!/usr/bin/env python3
"""
Experiment #030: 1d Donchian Breakout + 1w HMA Trend Filter + ADX + ATR Stop

Hypothesis: After 29 experiments, clearest patterns for 1d timeframe:
1. 1d is UNDERUTILIZED - only 3 attempts (#018, #024, this), both prior failed
2. #018 (RSI+BB) Sharpe=-2.131 - mean reversion FAILS on daily
3. #024 (KAMA+BB squeeze) Sharpe=-0.042 - adaptive MA alone insufficient
4. #023 (12h Donchian) Sharpe=0.137 - BREAKOUTS work on higher TFs
5. Daily needs SIMPLER logic - fewer filters = more trades (critical for 1d)
6. 1w HMA provides robust trend bias without overfitting (proven in #023)

This 1d strategy combines:

1. Donchian Channel (20): Classic breakout - above 20-day high = long, below low = short.
   Proven on higher timeframes (Turtle Trading legacy).

2. 1w HMA(21) trend filter: ONLY trade in direction of weekly trend.
   Price > 1w HMA = long only. Price < 1w HMA = short only.
   Single HTF filter (Rule 1 compliance).

3. ADX(14) > 15: Lower threshold for 1d (vs 18 on 12h) = more trades.
   Filters choppy ranges where breakouts fail.

4. ATR Trailing Stop: 3.0*ATR(14) - standard for daily timeframe.
   Protects from reversals without being too tight.

5. Discrete Sizing: 0.30 for strong (ADX>25), 0.20 for moderate (ADX>15).
   Conservative sizing protects from 2022-style crashes (77% BTC drop).

Why this should beat current best (Sharpe=0.137 on 12h):
- 1d TF naturally filters noise better than 12h
- Simpler logic = more trades (critical - 1d had too few trades before)
- Donchian breakouts PROVEN on higher timeframes
- Single 1w HMA filter = robust trend bias without complexity
- Lower ADX threshold (15 vs 18) = more entry opportunities

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 3.0 * ATR(14) trailing
Target trades: 15-40/year on 1d (optimal per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_adx_atr_v1"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    
    return upper.values, lower.values, middle.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.30  # ADX > 25 (strong trend)
    SIZE_MODERATE = 0.20  # ADX > 15 (moderate trend)
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_vs_1w = close[i] - hma_1w_aligned[i]
        
        bull_htf = price_vs_1w > 0  # Price above weekly HMA = bullish bias
        bear_htf = price_vs_1w < 0  # Price below weekly HMA = bearish bias
        
        # === ADX TREND STRENGTH ===
        adx_moderate = adx_14[i] > 15  # Trending market (lower threshold for 1d)
        adx_strong = adx_14[i] > 25  # Strong trend
        
        # === DONCHIAN BREAKOUT ===
        # Use previous bar's Donchian levels to avoid look-ahead
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: HTF bull + Donchian breakout + ADX confirmation
        if bull_htf and donchian_breakout_long and adx_moderate:
            if adx_strong and di_bull:
                new_signal = SIZE_STRONG  # All filters agree
            else:
                new_signal = SIZE_MODERATE  # Partial confirmation
        
        # SHORT ENTRY: HTF bear + Donchian breakout + ADX confirmation
        elif bear_htf and donchian_breakout_short and adx_moderate:
            if adx_strong and di_bear:
                new_signal = -SIZE_STRONG  # All filters agree
            else:
                new_signal = -SIZE_MODERATE  # Partial confirmation
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend reverses against position
            if position_side > 0 and bear_htf:
                trend_exit = True
            if position_side < 0 and bull_htf:
                trend_exit = True
            
            # Exit if price crosses Donchian middle against position
            if position_side > 0 and close[i] < donchian_middle[i]:
                trend_exit = True
            if position_side < 0 and close[i] > donchian_middle[i]:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
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