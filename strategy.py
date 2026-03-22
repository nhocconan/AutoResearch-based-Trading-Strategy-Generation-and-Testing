#!/usr/bin/env python3
"""
Experiment #002: 12h Connors RSI + Choppiness Regime with 1d Trend Filter

Hypothesis: Previous HMA-Donchian strategies fail in bear/range markets (2022 crash, 2025 test).
Connors RSI excels at mean reversion in choppy markets (75% win rate documented).
Choppiness Index detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend.
1d HMA(21) provides major trend bias to avoid counter-trend mean reversion.

Why this should work:
- Connors RSI catches oversold/overbought reversals (better than simple RSI)
- Choppiness filter avoids mean reversion during strong trends (whipsaw protection)
- 12h timeframe = 20-50 trades/year (manageable fee drag)
- 1d filter ensures we don't mean-revert against major trend

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_rsi_chop_regime_1d_filter_v1"
timeframe = "12h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of price change over lookback period
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: Short-term RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss
    streak_rs = streak_rs.replace([np.inf, -np.inf], np.nan).fillna(0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank of price changes
    price_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i]
        current = price_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100
        else:
            percent_rank.iloc[i] = 50.0
    
    percent_rank = percent_rank.fillna(50).values
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (mean reversion favorable)
    - CHOP < 38.2: Market is trending (trend following favorable)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR for each bar
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # Avoid division by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)  # Clip to valid range
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
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
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market - favor mean reversion
        is_trending = chop[i] < 38.2  # Trend market - favor trend following
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85  # Strong mean reversion short signal
        crsi_moderate_oversold = crsi[i] < 25  # Weaker long signal
        crsi_moderate_overbought = crsi[i] > 75  # Weaker short signal
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        # Primary: Choppy market + CRSI oversold + daily bias not bearish
        if is_choppy and crsi_oversold and not daily_bearish:
            new_signal = current_size
        # Secondary: Trending market + CRSI moderate oversold + daily bullish
        elif is_trending and crsi_moderate_oversold and daily_bullish:
            new_signal = current_size * 0.8
        # Tertiary: Very oversold regardless of regime (crash recovery)
        elif crsi[i] < 10:
            new_signal = current_size * 0.7
        
        # SHORT ENTRY
        # Primary: Choppy market + CRSI overbought + daily bias not bullish
        if is_choppy and crsi_overbought and not daily_bullish:
            new_signal = -current_size
        # Secondary: Trending market + CRSI moderate overbought + daily bearish
        elif is_trending and crsi_moderate_overbought and daily_bearish:
            new_signal = -current_size * 0.8
        # Tertiary: Very overbought regardless of regime
        elif crsi[i] > 90:
            new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 days on 12h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if crsi_moderate_oversold and not daily_bearish:
                new_signal = current_size * 0.5
            elif crsi_moderate_overbought and not daily_bullish:
                new_signal = -current_size * 0.5
        
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
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit when CRSI returns to neutral (50) - take profit on mean reversion
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 55:  # Long exit when CRSI > 55
                crsi_exit = True
            if position_side < 0 and crsi[i] < 45:  # Short exit when CRSI < 45
                crsi_exit = True
        
        # === REGIME CHANGE EXIT ===
        # Exit mean reversion position if market becomes strongly trending against us
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trending and daily_bearish:
                regime_exit = True
            if position_side < 0 and is_trending and daily_bullish:
                regime_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or regime_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals