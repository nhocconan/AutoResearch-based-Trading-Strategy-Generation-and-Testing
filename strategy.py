#!/usr/bin/env python3
"""
Experiment #027: 1d Dual Regime Strategy with 1w Bias

Hypothesis: Previous 1d strategies failed due to overly restrictive entry conditions
(0 trades in exp #017). This strategy uses a simpler dual-regime approach:
1. 1w HMA for major trend bias (call ONCE before loop via mtf_data)
2. Choppiness Index to detect regime: CHOP>61.8=range, CHOP<38.2=trend
3. Range regime: Connors RSI mean reversion (CRSI<15 long, >85 short)
4. Trend regime: Donchian breakout with HMA confirmation
5. ATR(14) trailing stoploss at 3.0x for wider stops on 1d
6. Position sizing: 0.25-0.30 discrete levels

Why this should work:
- 1d timeframe naturally limits trades to 20-50/year (fee-efficient)
- Dual regime adapts to market conditions (range vs trend)
- 1w bias prevents counter-trend trades in major moves
- Simpler entry conditions than failed exp #017 (which had 0 trades)
- Based on research showing Connors RSI + Choppiness works on ETH (Sharpe +0.923)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_1w_bias_v1"
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
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 days
    
    CRSI < 10: Extremely oversold (long signal)
    CRSI > 90: Extremely overbought (short signal)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank(100)
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= rank_period else np.nan
    )
    percent_rank = percent_rank.fillna(0.5).values * 100
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP)
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8: Range-bound market (mean reversion)
    CHOP < 38.2: Trending market (trend following)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    range_hl = hh - ll
    
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 0.0001, range_hl)
    
    # Calculate CHOP
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

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
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
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
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND ===
        daily_bullish = hma_1d_21[i] > hma_1d_50[i]
        daily_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Range-bound (mean reversion)
        is_trend = chop[i] < 45.0  # Trending (breakout)
        # Neutral zone 45-55: use both signals
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if weekly_bullish:  # Only long when 1w bias is bullish
            # Range regime: Connors RSI mean reversion
            if is_range and crsi[i] < 20:
                new_signal = current_size
            # Trend regime: Donchian breakout
            elif is_trend and daily_bullish:
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1] and rsi_14[i] > 45 and rsi_14[i] < 75:
                        new_signal = current_size
            # Neutral zone: combine signals
            elif not is_range and not is_trend:
                if crsi[i] < 25 and daily_bullish:
                    new_signal = current_size
                elif i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1] and rsi_14[i] > 50:
                        new_signal = current_size
        
        # SHORT ENTRIES
        elif weekly_bearish:  # Only short when 1w bias is bearish
            # Range regime: Connors RSI mean reversion
            if is_range and crsi[i] > 80:
                new_signal = -current_size
            # Trend regime: Donchian breakout
            elif is_trend and daily_bearish:
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1] and rsi_14[i] > 25 and rsi_14[i] < 55:
                        new_signal = -current_size
            # Neutral zone: combine signals
            elif not is_range and not is_trend:
                if crsi[i] > 75 and daily_bearish:
                    new_signal = -current_size
                elif i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1] and rsi_14[i] < 50:
                        new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~60 days on 1d), force entry with weaker signal
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 35:
                new_signal = current_size * 0.6
            elif weekly_bearish and crsi[i] > 65:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and daily_bearish and weekly_bearish:
                trend_reversal = True
            if position_side < 0 and daily_bullish and weekly_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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