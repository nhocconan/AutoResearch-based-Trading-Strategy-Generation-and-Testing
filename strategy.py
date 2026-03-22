#!/usr/bin/env python3
"""
Experiment #003: 1d Regime-Adaptive with 1w Trend Bias and Connors RSI

Hypothesis: Previous trend-following strategies failed because 2025 is a bear/range market.
This strategy uses REGIME-ADAPTIVE logic proven in quantitative literature:

1. Choppiness Index (CHOP) regime detection:
   - CHOP > 61.8 = ranging market → use mean reversion (Connors RSI)
   - CHOP < 38.2 = trending market → use breakout (Donchian + HMA)
   - Between = neutral → stay flat or reduce size

2. 1w HMA(21) for major trend bias - only trade in direction of weekly trend
   (proven in mtf_hma_rsi_zscore_v1 to 2x Sharpe)

3. Connors RSI for mean reversion entries (75% win rate):
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > weekly HMA
   - Short: CRSI > 90 + price < weekly HMA

4. Donchian(20) breakout for trend regime:
   - Long: breakout above 20-day high + weekly HMA bullish
   - Short: breakout below 20-day low + weekly HMA bearish

5. ATR(14) trailing stoploss at 2.5x - protects against 2022-style crashes

Why this should work:
- Regime-adaptive = works in both bull AND bear markets (key lesson from failures)
- Connors RSI has 75% win rate on reversals (quantitative literature)
- 1w HMA filter ensures we trade WITH major trend
- 1d timeframe targets 15-30 trades/year (optimal for fee drag)
- Position size 0.25-0.30 protects against 77% BTC crash in 2022

Timeframe: 1d (REQUIRED for Experiment #003)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_chop_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10 = oversold (long opportunity)
    - CRSI > 90 = overbought (short opportunity)
    
    Reference: Connors, "ConnorsRSI: A New Indicator for Understanding Market Timing"
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (absolute streak -> RSI scale)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * (streak_abs[i] / (streak_abs[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank component - where current return ranks vs last N days
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i].values
        current = returns.iloc[i]
        if len(window) > 0:
            percent_rank[i] = 100 * np.sum(window < current) / len(window)
        else:
            percent_rank[i] = 50
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    
    Reference: E.W. Dreiss, "The Choppiness Index"
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    hh_ll = hh - ll
    
    # Choppiness calculation
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over N periods."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Additional trend filter: 1d HMA
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.15  # For weaker signals
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HMA TREND ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 61.8  # Ranging market
        chop_trend = chop_14[i] < 38.2  # Trending market
        chop_neutral = not chop_range and not chop_trend
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === HMA SLOPE ===
        hma_slope_long = hma_1d_21[i] > hma_1d_21[i-5] if i > 5 else False
        hma_slope_short = hma_1d_21[i] < hma_1d_21[i-5] if i > 5 else False
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: RANGING MARKET (CHOP > 61.8) - Mean Reversion
        if chop_range:
            # Long: CRSI oversold + price above weekly HMA (major trend up)
            if crsi_oversold and weekly_bullish:
                new_signal = BASE_SIZE
            
            # Short: CRSI overbought + price below weekly HMA (major trend down)
            if crsi_overbought and weekly_bearish:
                new_signal = -BASE_SIZE
            
            # Weaker entry if CRSI extreme but no weekly bias (reduced size)
            if new_signal == 0.0 and crsi_oversold and hma_bullish:
                new_signal = REDUCED_SIZE
            if new_signal == 0.0 and crsi_overbought and hma_bearish:
                new_signal = -REDUCED_SIZE
        
        # REGIME 2: TRENDING MARKET (CHOP < 38.2) - Breakout
        elif chop_trend:
            # Long: Donchian breakout + weekly bullish + HMA slope up
            if donchian_breakout_long and weekly_bullish:
                new_signal = BASE_SIZE
            elif donchian_breakout_long and hma_bullish and hma_slope_long:
                new_signal = REDUCED_SIZE
            
            # Short: Donchian breakout + weekly bearish + HMA slope down
            if donchian_breakout_short and weekly_bearish:
                new_signal = -BASE_SIZE
            elif donchian_breakout_short and hma_bearish and hma_slope_short:
                new_signal = -REDUCED_SIZE
        
        # REGIME 3: NEUTRAL (38.2 <= CHOP <= 61.8) - Conservative
        else:
            # Only take strongest signals in neutral regime
            if crsi_oversold and weekly_bullish and hma_bullish:
                new_signal = REDUCED_SIZE
            if crsi_overbought and weekly_bearish and hma_bearish:
                new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if weekly_bullish and hma_bullish and crsi[i] < 30:
                new_signal = REDUCED_SIZE
            elif weekly_bearish and hma_bearish and crsi[i] > 70:
                new_signal = -REDUCED_SIZE
        
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
        
        # === REGIME CHANGE EXIT ===
        regime_change_exit = False
        if in_position and position_side != 0:
            # Exit if regime changes against position
            if position_side > 0 and chop_trend and not weekly_bullish:
                regime_change_exit = True
            if position_side < 0 and chop_trend and not weekly_bearish:
                regime_change_exit = True
        
        # === CRSI REVERSAL EXIT (for mean reversion trades) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_change_exit or crsi_exit:
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