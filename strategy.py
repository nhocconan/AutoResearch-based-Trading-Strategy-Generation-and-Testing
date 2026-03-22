#!/usr/bin/env python3
"""
Experiment #013: 1d Dual-Regime Strategy with Weekly Trend Bias

Hypothesis: Daily timeframe with weekly HTF filter provides optimal balance between
signal quality and trade frequency (target 20-50 trades/year). Previous regime strategies
failed due to overly strict entry conditions. This version uses:

1. Weekly HMA(21) for major trend bias (HTF via mtf_data)
2. Choppiness Index(14) for regime detection: >55=range, <45=trend
3. Connors RSI for mean reversion in range regime (entry <20 or >80)
4. Donchian(20) breakout for trend regime entries
5. ATR(14) stoploss at 2.5x for risk management
6. Position sizing: 0.25-0.30 discrete levels

Key improvements over #003 (failed 1d regime strategy):
- Looser CRSI thresholds (20/80 vs 15/85) for more trades
- Lower CHOP thresholds (55/45 vs 61.8/38.2) for better regime detection
- Frequency safeguard: force entry after 30 bars without trades
- Simpler exit logic (stoploss only, no complex trend reversal)
- Volatility-adjusted position sizing to reduce risk in high vol periods

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
Target trades: 20-50 per year (80-200 over 4-year train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_connors_donchian_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 55 = Range/Consolidation
    CHOP < 45 = Trending
    """
    atr = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    rsi_close = calculate_rsi(close, rsi_period)
    
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_avg_loss = np.where(streak_avg_loss == 0, 1e-10, streak_avg_loss)
    streak_rs = streak_avg_gain / streak_avg_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    returns = close_s.pct_change().values
    percent_rank = np.full(n, 50.0)
    
    for i in range(rank_period, n):
        if not np.isnan(returns[i]):
            window = returns[max(0, i-rank_period):i]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                percent_rank[i] = 100 * np.sum(window < returns[i]) / len(window)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

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
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === WEEKLY TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY HMA TREND ===
        daily_bullish = hma_1d_21[i] > hma_1d_50[i]
        daily_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
        else:
            atr_ratio = 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Connors RSI Mean Reversion (looser thresholds)
        if is_range:
            # LONG: CRSI < 20 (oversold) + weekly bias not strongly bearish
            if crsi[i] < 20 and not (weekly_bearish and daily_bearish):
                new_signal = current_size
            
            # SHORT: CRSI > 80 (overbought) + weekly bias not strongly bullish
            elif crsi[i] > 80 and not (weekly_bullish and daily_bullish):
                new_signal = -current_size
        
        # TREND REGIME: Donchian Breakout with trend confirmation
        elif is_trend:
            # LONG: Daily bullish + weekly bullish + Donchian breakout
            if daily_bullish and weekly_bullish:
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1]:
                        new_signal = current_size
            
            # SHORT: Daily bearish + weekly bearish + Donchian breakdown
            if daily_bearish and weekly_bearish:
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1]:
                        new_signal = -current_size
        
        # NEUTRAL REGIME (45-55): Use simpler HMA crossover with CRSI filter
        else:
            if daily_bullish and weekly_bullish and crsi[i] < 60:
                new_signal = current_size * 0.7
            elif daily_bearish and weekly_bearish and crsi[i] > 40:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~30 days on 1d), force entry with weaker conditions
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 45:
                new_signal = current_size * 0.5
            elif weekly_bearish and crsi[i] > 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # Apply stoploss
        if stoploss_triggered:
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