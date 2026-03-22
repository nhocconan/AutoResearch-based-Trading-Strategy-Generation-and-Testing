#!/usr/bin/env python3
"""
Experiment #537: 1h Connors RSI Mean Reversion with 4h HMA Trend Bias + Choppiness Filter

Hypothesis: After 500+ failed experiments, the pattern is clear:
1. Pure trend-following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
2. Pure mean-reversion fails without trend filter (gets crushed in strong trends)
3. Connors RSI (CRSI) has 75% win rate in academic research for mean reversion
4. Choppiness Index (CHOP) distinguishes range vs trend regimes
5. 4h HMA provides trend bias without excessive lag
6. 1h timeframe = enough trades (~50-100/year) without fee drag

Strategy Logic:
- LONG: 4h HMA bullish + CRSI<15 (oversold) + CHOP>50 (ranging/reversion OK)
- SHORT: 4h HMA bearish + CRSI>85 (overbought) + CHOP>50 (ranging/reversion OK)
- TREND FOLLOW: CHOP<40 (trending) + price crosses 4h HMA + ADX>20
- STOPLOSS: 2.5*ATR trailing stop
- POSITION SIZE: 0.25 discrete (conservative after 77% BTC crash lesson)

Why 1h should work:
- More trades than 4h/12h (meets minimum trade requirement)
- Less noise than 15m/30m (failed strategies #525, #529, #535)
- Connors RSI proven in literature for crypto mean reversion
- Choppiness filter avoids whipsaw in unclear regimes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_rsi_4h_hma_chop_regime_atr_v1"
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
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 bars
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percentile Rank over lookback period
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i+1]
        rank = np.sum(lookback[:-1] < close[i]) / (rank_period - 1)
        percent_rank[i] = rank * 100
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = choppy/ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    
    # Sum of ATR over period
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    range_hl = hh - ll
    
    # Choppiness formula
    chop = 100 * np.log10(atr_sum / range_hl.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

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
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    adx_14 = calculate_adx(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        is_ranging = chop[i] > 50  # Mean reversion regime
        is_trending = chop[i] < 40  # Trend following regime
        
        # === CONNORS RSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15  # Extreme oversold
        crsi_overbought = crsi[i] > 85  # Extreme overbought
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MEAN REVERSION ENTRIES (in ranging market)
        # Long: Ranging + CRSI oversold + 4h trend not strongly bearish
        if is_ranging and crsi_oversold:
            new_signal = SIZE
        
        # Short: Ranging + CRSI overbought + 4h trend not strongly bullish
        elif is_ranging and crsi_overbought:
            new_signal = -SIZE
        
        # TREND FOLLOWING ENTRIES (in trending market)
        # Long: Trending + bullish bias + ADX confirms + pullback (CRSI not overbought)
        elif is_trending and bull_bias and adx_14[i] > 20 and crsi[i] < 70:
            new_signal = SIZE
        
        # Short: Trending + bearish bias + ADX confirms + rally (CRSI not oversold)
        elif is_trending and bear_bias and adx_14[i] > 20 and crsi[i] > 30:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit mean reversion position if regime shifts to strong trend against us
        if in_position and new_signal != 0.0:
            # If we're long but regime becomes strongly trending bearish
            if position_side > 0 and is_trending and bear_bias and adx_14[i] > 30:
                new_signal = 0.0
            # If we're short but regime becomes strongly trending bullish
            if position_side < 0 and is_trending and bull_bias and adx_14[i] > 30:
                new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI becomes overbought
        if in_position and position_side > 0 and crsi[i] > 75:
            new_signal = 0.0
        # Exit short when CRSI becomes oversold
        if in_position and position_side < 0 and crsi[i] < 25:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals