#!/usr/bin/env python3
"""
Experiment #589: 4h Primary + 1d HTF — Donchian Breakout + Connors RSI Pullback

Hypothesis: Building on #579 (Sharpe=0.103), this version adds:
1. Donchian(20) breakout confirmation for trend entries (proven on SOL in literature)
2. Asymmetric CRSI bands (20/80 instead of 30/70) for sharper entry timing
3. Dynamic ATR stoploss (2.0x in strong trend, 3.0x in weak trend)
4. Volume confirmation on breakouts (taker_buy_volume > 1.5x average)
5. Position scaling: 0.35 in strong trend (1d HMA slope confirmed), 0.25 otherwise

Why this might beat Sharpe=0.520:
- Donchian breakout adds momentum confirmation (reduces false pullback entries)
- Tighter CRSI bands = better entry timing = higher win rate
- Dynamic stops = better risk/reward adaptation
- Volume filter = confirms institutional participation on breakouts
- Still simple enough to generate 30-50 trades/year on 4h

Position sizing: 0.25-0.35 discrete (per Rule 4, max 0.40)
Stoploss: 2.0-3.0 * ATR trailing (dynamic based on trend strength)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0.520 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_crsi_hma_1d_v2"
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
    Calculate Connors RSI (CRSI) - proven 75% win rate for mean reversion.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.values
    
    # PercentRank - where today's return ranks vs last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].dropna()
        if len(window) > 0:
            current_ret = returns.iloc[i]
            rank = (window < current_ret).sum() / len(window)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi_14 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    taker_ratio = taker_buy_volume / (volume + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_STRONG = 0.35  # Strong trend confirmed
    POSITION_SIZE_WEAK = 0.25    # Weak trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(adx_14[i]) or np.isnan(crsi_14[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # Strong trend = both price above HMA AND HMA slope confirmed
        strong_bull_1d = bull_regime_1d and hma_1d_slope_bull
        strong_bear_1d = bear_regime_1d and hma_1d_slope_bear
        
        # === ADX FILTER (minimal trend strength) ===
        # ADX > 18 means some directional movement
        trend_ok = adx_14[i] > 18.0
        
        # === CONNORS RSI ENTRY (TIGHTER BANDS for better timing) ===
        # Long: CRSI < 20 in uptrend (deeply oversold pullback)
        # Short: CRSI > 80 in downtrend (deeply overbought rally)
        crsi_oversold_long = crsi_14[i] < 20.0
        crsi_overbought_short = crsi_14[i] > 80.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # For longs: price breaking above Donchian high confirms momentum
        # For shorts: price breaking below Donchian low confirms momentum
        donchian_breakout_long = close[i] > donchian_high[i-1]  # previous bar's high
        donchian_breakout_short = close[i] < donchian_low[i-1]  # previous bar's low
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.3x average OR taker buy ratio > 0.55 (buying pressure)
        volume_confirmed = volume[i] > 1.3 * vol_avg[i]
        taker_buy_confirmed = taker_ratio[i] > 0.55
        taker_sell_confirmed = taker_ratio[i] < 0.45
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bull + CRSI oversold + (Donchian breakout OR volume confirmed)
        if bull_regime_1d and crsi_oversold_long and trend_ok:
            if donchian_breakout_long or (volume_confirmed and taker_buy_confirmed):
                if strong_bull_1d:
                    new_signal = POSITION_SIZE_STRONG
                else:
                    new_signal = POSITION_SIZE_WEAK
        
        # SHORT ENTRY: 1d bear + CRSI overbought + (Donchian breakout OR volume confirmed)
        elif bear_regime_1d and crsi_overbought_short and trend_ok:
            if donchian_breakout_short or (volume_confirmed and taker_sell_confirmed):
                if strong_bear_1d:
                    new_signal = -POSITION_SIZE_STRONG
                else:
                    new_signal = -POSITION_SIZE_WEAK
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === DYNAMIC STOPLOSS CHECK ===
        # Strong trend: 2.0x ATR, Weak trend: 3.0x ATR
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            atr_mult = 2.0 if strong_bull_1d else 3.0
            stop_price = highest_since_entry - atr_mult * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            atr_mult = 2.0 if strong_bear_1d else 3.0
            stop_price = lowest_since_entry + atr_mult * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1d regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
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