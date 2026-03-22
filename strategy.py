#!/usr/bin/env python3
"""
Experiment #484: 4h Multi-Signal Ensemble with Daily HMA Bias

Hypothesis: After 474 failed experiments, the critical insight is that 4h needs
MULTIPLE independent signal types that can trigger in different market conditions.
Single-condition strategies fail because they only work in specific regimes.

This strategy uses an ENSEMBLE approach:
1. DAILY HMA(21) for trend bias (via mtf_data helper)
2. THREE independent entry signals that can each trigger:
   - Signal A: Connors RSI < 20 (mean-reversion, works in ranges)
   - Signal B: Price > Donchian(20) high + ADX > 20 (trend breakout)
   - Signal C: RSI(7) < 30 + price > SMA(200) (dip buy in bull)
3. VOLUME confirmation on breakouts (avoid false moves)
4. ASYMMETRIC exits: bull=hold longer, bear=quick profits
5. ATR(14) trailing stop at 2.5x

Why this should work on 4h:
- Multiple entry types = more trades (critical for Sharpe calculation)
- Ensemble voting reduces false signals
- Daily HMA provides stable trend bias without whipsaw
- Should generate 30-60 trades/year per symbol

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ensemble_daily_hma_connors_donchian_rsi_atr_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    More sensitive than standard RSI for mean-reversion.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_close = (100 - (100 / (1 + rs))).values
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / streak_avg_loss.replace(0, np.inf)
    rsi_streak = (100 - (100 / (1 + streak_rs))).values
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio."""
    vol_s = pd.Series(volume)
    avg_vol = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / avg_vol.replace(0, np.inf)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(adx[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === ENSEMBLE ENTRY SIGNALS ===
        long_signals = 0
        short_signals = 0
        
        # Signal A: Connors RSI mean-reversion (works in all regimes)
        if crsi[i] < 25:
            long_signals += 1
        if crsi[i] > 75:
            short_signals += 1
        
        # Signal B: RSI(7) extreme + SMA200 filter
        if rsi_7[i] < 30 and close[i] > sma_200[i]:
            long_signals += 1
        if rsi_7[i] > 70 and close[i] < sma_200[i]:
            short_signals += 1
        
        # Signal C: Donchian breakout + ADX + volume (trend only)
        if adx[i] > 20:
            if close[i] > donchian_upper[i-1] and vol_ratio[i] > 1.0:
                if bull_regime:
                    long_signals += 1
            if close[i] < donchian_lower[i-1] and vol_ratio[i] > 1.0:
                if bear_regime:
                    short_signals += 1
        
        # Signal D: Pullback to SMA50 in trend
        if adx[i] > 18:
            if bull_regime and close[i] < sma_50[i] * 1.01 and close[i] > sma_50[i] * 0.99:
                long_signals += 1
            if bear_regime and close[i] > sma_50[i] * 0.99 and close[i] < sma_50[i] * 1.01:
                short_signals += 1
        
        # === DETERMINE SIGNAL ===
        new_signal = 0.0
        
        # Need at least 2 signals agreeing for entry (reduces false signals)
        if long_signals >= 2 and bull_regime:
            new_signal = SIZE
        elif short_signals >= 2 and bear_regime:
            new_signal = -SIZE
        # Single strong signal in strong trend
        elif long_signals >= 1 and bull_regime and adx[i] > 25 and crsi[i] < 20:
            new_signal = SIZE
        elif short_signals >= 1 and bear_regime and adx[i] > 25 and crsi[i] > 80:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TAKE PROFIT after 50 bars (reduce churn) ===
        if in_position and bars_in_trade > 50:
            if position_side > 0 and close[i] > entry_price * 1.08:
                new_signal = 0.0
            if position_side < 0 and close[i] < entry_price * 0.92:
                new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                bars_in_trade = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                bars_in_trade = 0
            else:
                bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                bars_in_trade = 0
        
        signals[i] = new_signal
    
    return signals