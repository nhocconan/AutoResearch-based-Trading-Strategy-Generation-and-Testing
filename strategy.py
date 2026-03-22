#!/usr/bin/env python3
"""
Experiment #001: 4h Regime-Adaptive Strategy with 1d/1w Trend Filter

Hypothesis: Previous trend-following strategies failed in 2022 crash and 2025 bear market
because they didn't adapt to market regime. This strategy uses:

1. CHOPPINESS INDEX (CHOP) for regime detection:
   - CHOP > 61.8 = Ranging market → Use mean-reversion (Connors RSI)
   - CHOP < 38.2 = Trending market → Use breakout (Donchian + HMA)
   - Between = No trade (avoid whipsaw)
   This is proven to improve Sharpe in bear/range markets (ETH +0.923 in research)

2. CONNORS RSI (CRSI) for mean-reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 + price > 1d HMA
   - Short when CRSI > 90 + price < 1d HMA
   - 75% win rate in research, catches reversals in bear rallies

3. DONCHIAN(20) BREAKOUT for trending entries:
   - Long when price breaks 20-bar high + price > 1d HMA
   - Short when price breaks 20-bar low + price < 1d HMA
   - Clean trend entry, proven on SOL (+0.782 Sharpe)

4. 1d HMA(21) Trend Filter via mtf_data helper:
   - Only long if price > 1d HMA, only short if price < 1d HMA
   - Prevents counter-trend failures

5. 1w HMA(21) Major Bias via mtf_data helper:
   - Increases position size when 4h and 1w trends align
   - Reduces size when they diverge (lower conviction)

6. ATR(14) Trailing Stop - 2.5x ATR for risk management
   - Signal → 0 when stopped out

Why this should work:
- Regime-adaptive = works in both bull (2021) and bear (2022, 2025) markets
- CHOP filter avoids trend-following whipsaw in ranges
- CRSI catches reversals that pure trend strategies miss
- 4h timeframe = 20-50 trades/year target (optimal for fee drag)
- Conservative sizing (0.25-0.30) protects against 77% crashes
- Multiple entry types = more trades (avoids 0-trade failure)

Timeframe: 4h (REQUIRED for Experiment #001)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_crsi_donchian_1d_1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = Ranging/consolidating market
    - CHOP < 38.2 = Trending market
    - Between = Transition/neutral
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    price_range = hh - ll
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean-reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) - short-term momentum
    2. RSI_Streak(2) - streak duration RSI (consecutive up/down days)
    3. PercentRank(100) - where current return sits in recent distribution
    
    Entry signals:
    - CRSI < 10 = Oversold (long opportunity)
    - CRSI > 90 = Overbought (short opportunity)
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Component 2: RSI Streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    rs_streak = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Component 3: PercentRank of returns
    returns = close_s.pct_change() * 100
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50,
        raw=False
    )
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    crsi = crsi.replace([np.inf, -np.inf], np.nan)
    
    return crsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel highs and lows."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    donchian_high = high_s.rolling(window=period, min_periods=period).max().values
    donchian_low = low_s.rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    return donchian_high, donchian_low, donchian_mid

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
    
    # Calculate 4h indicators
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, 14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(donchian_high[i]):
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = Ranging, CHOP < 38.2 = Trending
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- MEAN REVERSION ENTRIES (Ranging Regime) ---
        if is_ranging:
            # Connors RSI mean reversion
            crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 for more trades
            crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90
            
            # Long: CRSI oversold + daily bullish bias
            if crsi_oversold and daily_bullish:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short: CRSI overbought + daily bearish bias
            elif crsi_overbought and daily_bearish:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        # --- TREND FOLLOWING ENTRIES (Trending Regime) ---
        elif is_trending:
            # Detect Donchian breakouts
            donchian_breakout_long = False
            donchian_breakout_short = False
            
            if i > 0:
                if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                    donchian_breakout_long = True
                if close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                    donchian_breakout_short = True
            
            # Also allow continuation entries
            above_donchian = close[i] > donchian_high[i]
            below_donchian = close[i] < donchian_low[i]
            
            # Long: Donchian breakout + daily bullish
            if (donchian_breakout_long or above_donchian) and daily_bullish:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short: Donchian breakout + daily bearish
            elif (donchian_breakout_short or below_donchian) and daily_bearish:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        # --- NEUTRAL REGIME ---
        # Only allow low-conviction trades if no position and long since last trade
        elif not in_position:
            bars_since_last_trade = i - last_trade_bar
            if bars_since_last_trade > 60:  # ~10 days on 4h
                # Allow weak mean-reversion entries
                if crsi[i] < 10 and daily_bullish:
                    new_signal = LOW_CONV_SIZE
                elif crsi[i] > 90 and daily_bearish:
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
        
        # === CRSI EXIT (for mean-reversion trades) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 30:
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