#!/usr/bin/env python3
"""
Experiment #599: 4h Primary + 1d HTF — Dual Regime (Chop/Trend) + HMA + CRSI

Hypothesis: Building on #591/#594 success (4h chop regime + HMA/KAMA), this strategy
uses 4h as primary timeframe with 1d HTF for stronger trend confirmation. Key innovations:
1. Choppiness Index regime switch (CHOP>55 mean revert, CHOP<45 trend follow)
2. Connors RSI for mean reversion entries (proven 75% win rate in literature)
3. 1d HMA(50) for macro trend bias (stronger than 4h HMA alone)
4. ADX filter for trend regime confirmation
5. ATR trailing stoploss (2.5x) for risk management

Why this might beat Sharpe=0.520:
- 4h TF proven successful (#591 Sharpe=0.423, #594 Sharpe=0.465)
- 1d HTF adds macro trend filter without over-filtering
- Dual regime adapts to market conditions (chop vs trend)
- CRSI extremes capture reversals better than standard RSI
- Position size 0.30 balances return vs drawdown

Position sizing: 0.30 discrete (appropriate for 4h per Rule 10)
Target: >=20 trades/symbol train, >=3 test, Sharpe > 0 all symbols
Trade frequency: 20-50/year on 4h (use regime + ADX to control)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_1d_v2"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    CRSI < 15 = extremely oversold (long signal)
    CRSI > 85 = extremely overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - fast RSI
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_s = pd.Series(streak_gain)
    streak_loss_s = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100)
    price_change = close_s.diff()
    percent_rank = pd.Series(np.zeros(n))
    
    for i in range(rank_period, n):
        window = price_change.iloc[i-rank_period:i]
        current = price_change.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    percent_rank = percent_rank.fillna(50.0).values
    
    # CRSI composite
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 55 = choppy/range market (mean reversion works)
    CHOP < 45 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for macro trend bias
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_100 = calculate_hma(df_1d['close'].values, period=100)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_100_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_100)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # 4h HMA for local trend
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    # Volume SMA for confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
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
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(hma_1d_100_aligned[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(vol_sma20[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop_regime = chop_14[i] > 55.0
        is_trend_regime = chop_14[i] < 45.0
        
        # === 1D MACRO TREND BIAS ===
        bull_bias_1d = close[i] > hma_1d_50_aligned[i]
        bear_bias_1d = close[i] < hma_1d_50_aligned[i]
        strong_bull_1d = hma_1d_50_aligned[i] > hma_1d_100_aligned[i]
        strong_bear_1d = hma_1d_50_aligned[i] < hma_1d_100_aligned[i]
        
        # === 4H LOCAL TREND ===
        bull_bias_4h = close[i] > hma_4h_21[i]
        bear_bias_4h = close[i] < hma_4h_21[i]
        hma_4h_slope_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_slope_bear = hma_4h_21[i] < hma_4h_50[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.7 * vol_sma20[i]
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 18.0
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        if volume_confirmed:
            # --- CHOP REGIME: Mean Reversion (CRSI extremes) ---
            if is_chop_regime:
                # Long: CRSI < 20 (oversold) + 1d not strongly bearish
                if crsi[i] < 20.0 and not (bear_bias_1d and strong_bear_1d):
                    new_signal = POSITION_SIZE
                
                # Short: CRSI > 80 (overbought) + 1d not strongly bullish
                elif crsi[i] > 80.0 and not (bull_bias_1d and strong_bull_1d):
                    new_signal = -POSITION_SIZE
            
            # --- TREND REGIME: Trend Following (CRSI pullback + ADX) ---
            elif is_trend_regime:
                # Long: CRSI < 40 (pullback) + ADX strong + 1d bull + 4h bull
                if crsi[i] < 40.0 and strong_trend and bull_bias_1d and bull_bias_4h:
                    new_signal = POSITION_SIZE
                
                # Short: CRSI > 60 (pullback) + ADX strong + 1d bear + 4h bear
                elif crsi[i] > 60.0 and strong_trend and bear_bias_1d and bear_bias_4h:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        if in_position and position_side > 0:
            # Exit long on strong bear bias flip
            if bear_bias_1d and strong_bear_1d and crsi[i] > 50.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on strong bull bias flip
            if bull_bias_1d and strong_bull_1d and crsi[i] < 50.0:
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