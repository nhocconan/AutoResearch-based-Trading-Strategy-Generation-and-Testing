#!/usr/bin/env python3
"""
Experiment #523: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 468 failed strategies, return to proven 1d primary timeframe with
regime-adaptive logic. The current best (Sharpe=0.435) uses 1d+1w, so this TF combo works.

Key insights from research:
- Choppiness Index > 61.8 = range market (mean revert with Connors RSI)
- Choppiness Index < 38.2 = trending market (trend follow with HMA/Donchian)
- Connors RSI < 10 + price > SMA200 = high-probability long (75% win rate)
- 1w HMA for major trend direction prevents counter-trend trades

This strategy uses:
1. Choppiness Index(14) for regime detection (range vs trend)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean reversion
3. HMA(21/50) on 1d for trend entries
4. 1w HMA(21) for major trend filter
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete position sizing (0.30) to minimize fee churn

Why this might work:
- Regime-adaptive: different logic for chop vs trend (proven on ETH Sharpe +0.923)
- Connors RSI catches oversold bounces in bull, overbought dumps in bear
- 1w trend filter prevents fighting the major trend
- 1d TF targets 20-40 trades/year (optimal for fee/trade ratio)
- Simple enough to generate trades, complex enough to filter noise

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_connors_hma_1w_regime_v1"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Apply RSI to streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank of daily returns
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= rank_period else np.nan
    )
    percent_rank = percent_rank * 100.0  # Scale to 0-100
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # HMA for trend detection
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    
    # SMA for trend filter
    sma_200 = calculate_sma(close, 200)
    
    # Choppiness Index for regime detection
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Connors RSI for mean reversion entries
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Standard RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    HALF_POSITION = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8  # Range market - mean revert
        trending_regime = chop_14[i] < 38.2  # Trending market - trend follow
        neutral_regime = not choppy_regime and not trending_regime
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND (secondary filter) ===
        hma_bull = hma_1d_21[i] > hma_1d_50[i]
        hma_bear = hma_1d_21[i] < hma_1d_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # --- CHOPPY REGIME: Mean Reversion with Connors RSI ---
        if choppy_regime:
            # Long: CRSI extremely oversold + above SMA200 + 1w bull
            if crsi[i] < 15.0 and price_above_sma200 and bull_regime:
                new_signal = POSITION_SIZE
            # Long: CRSI oversold + 1d HMA bull
            elif crsi[i] < 20.0 and hma_bull and bull_regime:
                new_signal = POSITION_SIZE
            # Short: CRSI extremely overbought + below SMA200 + 1w bear
            elif crsi[i] > 85.0 and price_below_sma200 and bear_regime:
                new_signal = -POSITION_SIZE
            # Short: CRSI overbought + 1d HMA bear
            elif crsi[i] > 80.0 and hma_bear and bear_regime:
                new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following with HMA ---
        elif trending_regime:
            # Long: HMA crossover up + 1w bull + RSI not overbought
            hma_cross_up = (hma_1d_21[i] > hma_1d_50[i]) and (hma_1d_21[i-1] <= hma_1d_50[i-1])
            if hma_cross_up and bull_regime and rsi_14[i] < 70.0:
                new_signal = POSITION_SIZE
            # Long: Pullback to HMA21 in uptrend + 1w bull
            elif hma_bull and bull_regime and close[i] < hma_1d_21[i] * 1.02 and rsi_14[i] < 50.0:
                new_signal = HALF_POSITION
            # Short: HMA crossover down + 1w bear + RSI not oversold
            hma_cross_down = (hma_1d_21[i] < hma_1d_50[i]) and (hma_1d_21[i-1] >= hma_1d_50[i-1])
            if hma_cross_down and bear_regime and rsi_14[i] > 30.0:
                new_signal = -POSITION_SIZE
            # Short: Bounce to HMA21 in downtrend + 1w bear
            elif hma_bear and bear_regime and close[i] > hma_1d_21[i] * 0.98 and rsi_14[i] > 50.0:
                new_signal = -HALF_POSITION
        
        # --- NEUTRAL REGIME: Conservative entries only ---
        else:
            # Only enter on extreme Connors RSI with trend confirmation
            if crsi[i] < 10.0 and bull_regime and hma_bull:
                new_signal = HALF_POSITION
            elif crsi[i] > 90.0 and bear_regime and hma_bear:
                new_signal = -HALF_POSITION
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on regime flip to bear or extreme overbought
        if in_position and position_side > 0:
            if bear_regime and hma_bear:
                new_signal = 0.0
            elif rsi_14[i] > 80.0:  # Extreme overbought
                new_signal = 0.0
            elif crsi[i] > 85.0:  # Connors extreme
                new_signal = 0.0
        
        # Exit short on regime flip to bull or extreme oversold
        if in_position and position_side < 0:
            if bull_regime and hma_bull:
                new_signal = 0.0
            elif rsi_14[i] < 20.0:  # Extreme oversold
                new_signal = 0.0
            elif crsi[i] < 15.0:  # Connors extreme
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals