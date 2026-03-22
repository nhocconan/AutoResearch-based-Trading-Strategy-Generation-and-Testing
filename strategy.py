#!/usr/bin/env python3
"""
Experiment #027: 1d Connors RSI + Choppiness Regime + 1w HMA Trend Filter

Hypothesis: Previous 1d strategies failed because they used pure trend-following
which gets destroyed in bear/range markets (2022 crash, 2025 bear). This strategy
uses Connors RSI for mean-reversion entries (75% win rate in literature) with
Choppiness Index to detect regime (range vs trend), and 1w HMA for major trend bias.

Key components:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 (oversold)
   - Short when CRSI > 90 (overbought)
2. Choppiness Index (CHOP 14) for regime detection
   - CHOP > 61.8 = range market (enable mean reversion)
   - CHOP < 38.2 = trending market (reduce mean reversion, wait for pullback)
3. 1w HMA(21) for major trend bias via mtf_data helper
   - Price > 1w HMA = long bias only
   - Price < 1w HMA = short bias only
4. ATR(14) trailing stoploss at 2.5x ATR

Why this should work:
- Daily timeframe = 20-50 trades/year (fee drag manageable)
- Connors RSI proven for mean reversion in crypto
- Choppiness filter avoids mean reversion in strong trends
- 1w HMA ensures we trade with major trend direction
- Conservative sizing (0.25-0.30) limits drawdown

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_chop_regime_1w_hma_atr_v1"
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

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI of streak (consecutive up/down days)
    # Streak: count consecutive days with same direction
    direction = np.sign(np.diff(close, prepend=close[0]))
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if direction[i] == direction[i-1]:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = streak_avg_gain / streak_avg_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank of daily returns
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(period_rank, n):
        window = returns.iloc[max(0, i-period_rank):i]
        if len(window) > 0:
            current_return = returns.iloc[i]
            rank = (window < current_return).sum() / len(window)
            percent_rank.iloc[i] = rank * 100
    
    percent_rank = percent_rank.fillna(50).values
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR(1), period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    # Calculate True Range for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    choppiness = calculate_choppiness_index(high, low, close, 14)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(choppiness[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPY INDEX REGIME ===
        is_range = choppiness[i] > 61.8  # Range market - enable mean reversion
        is_trend = choppiness[i] < 38.2  # Trending market - be more selective
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Very oversold
        crsi_overbought = crsi[i] > 85  # Very overbought
        crsi_extreme_oversold = crsi[i] < 10  # Extremely oversold
        crsi_extreme_overbought = crsi[i] > 90  # Extremely overbought
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Mean reversion in range market with weekly bullish bias
        # OR pullback in trending market with weekly bullish bias
        if weekly_bullish:
            if is_range and crsi_oversold:
                # Range market + oversold = strong mean reversion long
                new_signal = current_size
            elif is_trend and crsi_extreme_oversold:
                # Trending market + extremely oversold = pullback long
                new_signal = current_size * 0.7  # Smaller size in trend
            elif crsi_extreme_oversold:
                # Extremely oversold regardless of regime (with weekly bias)
                new_signal = current_size * 0.7
        
        # SHORT ENTRY: Mean reversion in range market with weekly bearish bias
        # OR rally in trending market with weekly bearish bias
        if weekly_bearish:
            if is_range and crsi_overbought:
                # Range market + overbought = strong mean reversion short
                new_signal = -current_size
            elif is_trend and crsi_extreme_overbought:
                # Trending market + extremely overbought = rally short
                new_signal = -current_size * 0.7  # Smaller size in trend
            elif crsi_extreme_overbought:
                # Extremely overbought regardless of regime (with weekly bias)
                new_signal = -current_size * 0.7
        
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
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI becomes overbought
        # Exit short when CRSI becomes oversold
        crsi_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi_overbought:
                crsi_reversal = True
            if position_side < 0 and crsi_oversold:
                crsi_reversal = True
        
        # === WEEKLY TREND REVERSAL EXIT ===
        weekly_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                weekly_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
                weekly_reversal = True
        
        # Apply stoploss or reversal
        if stoploss_triggered or crsi_reversal or weekly_reversal:
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
            # else: same direction, maintain position
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