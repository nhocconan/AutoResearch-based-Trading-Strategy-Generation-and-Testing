#!/usr/bin/env python3
"""
Experiment #544: 4h Connors RSI Mean Reversion with Daily HMA Bias + Choppiness Filter

Hypothesis: After 500+ failed experiments, the clearest pattern is:
1. Pure trend-following (EMA, Supertrend, Donchian) FAILS on BTC/ETH (too much whipsaw)
2. Mean reversion WORKS better in bear/range markets (2022 crash, 2025 bear)
3. Connors RSI has 75% win rate in literature for mean reversion entries
4. Choppiness Index detects range vs trend regime (CHOP>50 = range = mean revert)
5. Daily HMA provides longer-term bias to avoid counter-trend mean reversion
6. 4h timeframe balances signal frequency vs noise (more trades than 12h/1d)

Why this should work:
- CRSI<15 is oversold but not impossible (generates trades)
- CRSI>85 is overbought but achievable in rallies
- CHOP filter avoids mean reversion during strong trends (major failure mode)
- 1d HMA bias prevents shorting in bull markets / longing in bear markets
- 2.5*ATR stoploss protects against sustained moves against position
- Position size 0.25 = manageable drawdown during 2022-style crashes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_connors_rsi_1d_hma_chop_regime_meanrev_atr_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Measures consecutive up/down days as percentile.
    """
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    # Count consecutive up/down streaks
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            if delta.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta.iloc[i] < 0:
            if delta.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like score (absolute streak strength)
    streak_s = pd.Series(streak)
    up_streak = streak_s.where(streak_s > 0, 0.0)
    down_streak = -streak_s.where(streak_s < 0, 0.0)
    
    # Calculate streak RSI
    avg_up = up_streak.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_down = down_streak.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_up / avg_down.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + rs))
    
    return rsi_streak.values

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Measures current price change vs historical distribution.
    """
    close_s = pd.Series(close)
    returns = close_s.pct_change() * 100
    
    percent_rank = pd.Series(np.zeros(len(close)), index=close_s.index)
    
    for i in range(period, len(close)):
        window = returns.iloc[i-period:i]
        current = returns.iloc[i]
        if len(window) > 0 and not np.isnan(current):
            rank = (window < current).sum() / len(window) * 100
            percent_rank.iloc[i] = rank
    
    return percent_rank.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Range: 0-100, oversold < 10-15, overbought > 85-90
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Range: 0-100
    CHOP > 61.8 = choppy/range market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(hh_ll / tr_sum) / np.log10(period)
    
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 50 = range/choppy (favor mean reversion)
        # CHOP < 40 = trending (avoid mean reversion)
        range_market = chop_14[i] > 50
        
        # === ADX FILTER (avoid strong trends for mean reversion) ===
        # ADX < 30 = not strongly trending (better for mean reversion)
        not_strong_trend = adx_14[i] < 30
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        oversold = crsi[i] < 15
        overbought = crsi[i] > 85
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: CRSI oversold + daily bullish bias + range market OR weak trend
        if oversold and bull_bias and (range_market or not_strong_trend):
            new_signal = SIZE
        
        # Short: CRSI overbought + daily bearish bias + range market OR weak trend
        if overbought and bear_bias and (range_market or not_strong_trend):
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
        
        # === MEAN REVERSION EXIT (CRSI returns to neutral) ===
        # Exit long when CRSI > 50 (returned to neutral)
        # Exit short when CRSI < 50 (returned to neutral)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 50:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 50:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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