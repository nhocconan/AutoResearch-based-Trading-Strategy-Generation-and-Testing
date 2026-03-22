#!/usr/bin/env python3
"""
Experiment #002: 12h Dual Regime Strategy with Choppiness + Connors RSI + 1d/1w HMA Filter

Hypothesis: Previous 4h breakout strategies failed because they whipsawed in 2022 bear market.
This 12h strategy uses a DUAL REGIME approach that adapts to market conditions:

1. CHOPPINESS INDEX (14) - Regime Detection
   - CHOP > 61.8 = Range/Chop market → Mean Reversion mode
   - CHOP < 38.2 = Trending market → Trend Following mode
   - This is CRITICAL for 2022-2025 bear/range markets where pure trend fails

2. CONNORS RSI (CRSI) - Entry Timing
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold in range) or CRSI < 30 + trend bullish
   - Short entry: CRSI > 85 (overbought in range) or CRSI > 70 + trend bearish
   - 75% win rate in literature, works well in crypto mean-reversion

3. 1d HMA(21) Trend Filter - via mtf_data helper
   - Only long if price > 1d HMA (bullish daily trend)
   - Only short if price < 1d HMA (bearish daily trend)
   - Prevents counter-trend trades that fail in strong trends

4. 1w HMA(21) Major Bias - via mtf_data helper
   - Increases size when 12h, 1d, 1w all align (high conviction)
   - Reduces size when timeframes diverge

5. ATR(14) Trailing Stop - 2.5x ATR for risk management
   - Signal → 0 when stopped out
   - Protects against 2022-style crashes

Why 12h timeframe:
- 20-50 trades/year target (optimal fee drag: 1-2.5%)
- Less noise than 4h, more signals than 1d
- Works for both bull (2021) and bear (2022, 2025) markets
- Dual regime adapts to changing conditions

Position sizing: 0.25 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 12h (REQUIRED for Experiment #002)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_regime_1d_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending.
    CHOP > 61.8 = Range/Chop
    CHOP < 38.2 = Trending
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

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
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI Streak component.
    Measures consecutive up/down days.
    """
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    # Streak: consecutive up (+) or down (-) days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    abs_streak = np.abs(streak_s)
    
    # RSI of streak values
    gain_streak = streak_s.where(streak_s > 0, 0.0)
    loss_streak = -streak_s.where(streak_s < 0, 0.0)
    
    avg_gain = gain_streak.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss_streak.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    streak_rsi = 100 - (100 / (1 + rs))
    streak_rsi = streak_rsi.replace([np.inf, -np.inf], np.nan)
    
    return streak_rsi.values

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Where current close sits in recent distribution (0-100).
    """
    close_s = pd.Series(close)
    
    def pct_rank(x):
        if len(x) < 2 or x.max() == x.min():
            return 50.0
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
    
    pr = close_s.rolling(window=period, min_periods=period).apply(pct_rank, raw=False)
    pr = pr.replace([np.inf, -np.inf], np.nan)
    
    return pr.values

def calculate_connors_rsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_3 = calculate_rsi(close, period=3)
    streak_rsi = calculate_streak_rsi(close, period=2)
    pct_rank = calculate_percent_rank(close, period=100)
    
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Simple moving average for additional filter
    close_s = pd.Series(close)
    sma_50 = close_s.rolling(window=50, min_periods=50).mean().values
    sma_200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):  # Start after 200 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = Range/Chop → Mean Reversion mode
        # CHOP < 38.2 = Trending → Trend Following mode
        # 38.2 - 61.8 = Transition zone
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20  # Strong oversold
        crsi_very_oversold = crsi[i] < 10  # Extreme oversold
        crsi_overbought = crsi[i] > 80  # Strong overbought
        crsi_very_overbought = crsi[i] > 90  # Extreme overbought
        
        # Moderate extremes for more trades
        crsi_moderate_oversold = crsi[i] < 30
        crsi_moderate_overbought = crsi[i] > 70
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        long_score = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE: Buy oversold in range
            if crsi_very_oversold:
                long_score += 4.0
            elif crsi_oversold:
                long_score += 3.0
            elif crsi_moderate_oversold:
                long_score += 2.0
            
            # Add points for trend alignment
            if daily_bullish:
                long_score += 1.5
            if weekly_bullish:
                long_score += 1.0
            if above_sma50:
                long_score += 0.5
            
            # Entry threshold for choppy regime
            if long_score >= 4.0:
                if weekly_bullish and daily_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
        
        elif is_trending:
            # TREND FOLLOWING MODE: Buy pullbacks in uptrend
            if daily_bullish:
                long_score += 3.0
                if crsi_moderate_oversold:  # Pullback entry
                    long_score += 2.0
                if weekly_bullish:
                    long_score += 2.0
                if above_sma200:
                    long_score += 1.0
                
                if long_score >= 5.0:
                    if weekly_bullish:
                        new_signal = HIGH_CONV_SIZE
                    else:
                        new_signal = BASE_SIZE
        
        else:
            # TRANSITION ZONE: Conservative entries
            if crsi_oversold and daily_bullish and above_sma50:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY
        short_score = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE: Sell overbought in range
            if crsi_very_overbought:
                short_score += 4.0
            elif crsi_overbought:
                short_score += 3.0
            elif crsi_moderate_overbought:
                short_score += 2.0
            
            # Add points for trend alignment
            if daily_bearish:
                short_score += 1.5
            if weekly_bearish:
                short_score += 1.0
            if not above_sma50:
                short_score += 0.5
            
            # Entry threshold for choppy regime
            if short_score >= 4.0:
                if weekly_bearish and daily_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND FOLLOWING MODE: Sell rallies in downtrend
            if daily_bearish:
                short_score += 3.0
                if crsi_moderate_overbought:  # Rally entry
                    short_score += 2.0
                if weekly_bearish:
                    short_score += 2.0
                if not above_sma200:
                    short_score += 1.0
                
                if short_score >= 5.0:
                    if weekly_bearish:
                        new_signal = -HIGH_CONV_SIZE
                    else:
                        new_signal = -BASE_SIZE
        
        else:
            # TRANSITION ZONE: Conservative entries
            if crsi_overbought and daily_bearish and not above_sma50:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if crsi_moderate_oversold and daily_bullish:
                new_signal = LOW_CONV_SIZE
            elif crsi_moderate_overbought and daily_bearish:
                new_signal = -LOW_CONV_SIZE
        
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
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long if CRSI becomes overbought (take profit)
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short if CRSI becomes oversold (take profit)
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit:
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