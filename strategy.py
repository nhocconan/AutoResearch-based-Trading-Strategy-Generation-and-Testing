#!/usr/bin/env python3
"""
Experiment #043: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Daily timeframe with weekly trend filter reduces fee drag while
capturing major moves. Dual regime adapts to market conditions:
1. CHOPPY regime (Choppiness > 55): Mean reversion via Connors RSI extremes
2. TREND regime (Choppiness < 45): Trend following via Donchian breakout + 1w HMA

Why this should work:
- 1d timeframe = 10-30 trades/year (minimal fee drag)
- 1w HMA filters out counter-trend trades in major trends
- Connors RSI has 75% win rate for mean reversion at extremes
- Choppiness Index prevents trend-following in ranges (major failure mode)
- Donchian(20) breakout captures sustained moves
- ATR(14) trailing stop at 2.5x protects capital
- Discrete sizing (0.25/0.30) minimizes signal churn

Timeframe: 1d (REQUIRED per experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-35/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_donchian_1w_v3"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Component 3: Percent Rank
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else 50
    )
    percent_rank = percent_rank.fillna(50).values
    
    # Connors RSI
    connors_rsi = (rsi_3 + streak_rsi + percent_rank) / 3
    return connors_rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
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
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Additional trend filter: 1d HMA
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = close[i] > hma_1d_21[i] and hma_1d_21[i] > hma_1d_50[i]
        trend_1d_bearish = close[i] < hma_1d_21[i] and hma_1d_21[i] < hma_1d_50[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion preferred)
        # CHOP < 45 = trend (trend following preferred)
        # 45-55 = transition (reduce size or stay flat)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        is_transition = 45 <= chop[i] <= 55
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = extremely oversold (long opportunity in range)
        # CRSI > 85 = extremely overbought (short opportunity in range)
        crsi_oversold = connors_rsi[i] < 15
        crsi_overbought = connors_rsi[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        # Break above upper = bullish breakout
        # Break below lower = bearish breakout
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === VOLUME MOMENTUM (optional confirmation) ===
        # Use close-close momentum as proxy
        mom_5 = (close[i] / close[i-5] - 1) * 100 if i >= 5 else 0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transition regime
        if is_transition:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION ENTRIES (in range regime)
        # Long: Range + CRSI oversold + 1w trend not strongly bearish
        if is_range and crsi_oversold:
            if not trend_1w_bearish or connors_rsi[i] < 10:  # Very extreme overrides trend
                new_signal = current_size
        
        # Short: Range + CRSI overbought + 1w trend not strongly bullish
        if is_range and crsi_overbought:
            if not trend_1w_bullish or connors_rsi[i] > 90:  # Very extreme overrides trend
                new_signal = -current_size
        
        # TREND FOLLOWING ENTRIES (in trend regime)
        # Long: Trend + Donchian breakout + 1w bullish confirmation
        if is_trend and donchian_breakout_long:
            if trend_1w_bullish and trend_1d_bullish:
                new_signal = current_size
        
        # Short: Trend + Donchian breakout + 1w bearish confirmation
        if is_trend and donchian_breakout_short:
            if trend_1w_bearish and trend_1d_bearish:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Weaker mean reversion entry
            if is_range and connors_rsi[i] < 20:
                new_signal = current_size * 0.6
            elif is_range and connors_rsi[i] > 80:
                new_signal = -current_size * 0.6
            # Weaker trend entry
            elif is_trend and trend_1w_bullish and mom_5 > 5:
                new_signal = current_size * 0.6
            elif is_trend and trend_1w_bearish and mom_5 < -5:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_change_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trend and connors_rsi[i] > 70:
                # Long in trend regime but RSI overbought = take profit
                regime_change_exit = True
            if position_side < 0 and is_trend and connors_rsi[i] < 30:
                # Short in trend regime but RSI oversold = take profit
                regime_change_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1w_bearish and connors_rsi[i] > 60:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and connors_rsi[i] < 40:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_change_exit or trend_reversal:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, maintain position
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