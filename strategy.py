#!/usr/bin/env python3
"""
Experiment #548: 30m Primary + 4h/1d HTF — Ultra-Selective Confluence Strategy

Hypothesis: After 480+ failed strategies, the clearest pattern for lower TF is:
- 30m strategies FAIL when they trade too much (#538 Sharpe=-0.658, #545 Sharpe=-1.003)
- The #1 failure mode: >200 trades/year → fee drag destroys all edge
- Solution: Use HTF (1d/4h) for DIRECTION, 30m only for ENTRY TIMING
- This gives HTF trade frequency (30-80/year) with 30m execution precision

This strategy uses TRIPLE CONFLUENCE (all 3 must agree):
1. 1d HMA(50) slope = PRIMARY TREND DIRECTION (only trade with 1d trend)
2. 4h Choppiness Index < 45 = TRENDING REGIME (avoid choppy markets)
3. 30m Connors RSI extreme (<12 or >88) + volume spike = ENTRY TRIGGER

Additional filters to minimize trades:
- Volume > 1.2x 20-bar average (confirm conviction)
- Price > 1d HMA(50) for longs, < for shorts (trend confirmation)
- 4h ADX > 20 (trend strength confirmation)
- Position size: 0.20 (smaller for lower TF per Rule 4)
- ATR(14) 2.5x trailing stoploss

Why this might beat Sharpe=0.435:
- Triple confluence = very few trades (target 40-70/year)
- 1d trend filter prevents counter-trend losses (key failure in 2022)
- 4h regime filter avoids mean-reversion whipsaw in trends
- 30m CRSI extreme catches precise entry within HTF trend
- Small position size (0.20) limits drawdown on inevitable losses

Position sizing: 0.20 base (discrete per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_triple_confluence_crsi_4h1d_v1"
timeframe = "30m"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - research-backed mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry: CRSI < 10-15 (oversold) for long, CRSI > 85-90 (overbought) for short
    Research shows 75% win rate with trend filter.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streaks
    direction = np.sign(np.diff(close, prepend=close[0]))
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if direction[i] == direction[i-1]:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection indicator.
    
    CHOP > 61.8: Market is choppy/ranging (avoid trading)
    CHOP < 38.2: Market is trending (trade with trend)
    38.2 < CHOP < 61.8: Transition (reduced conviction)
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for PRIMARY trend direction
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Calculate 4h HTF indicators for regime
    chop_4h = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    adx_4h = calculate_adx(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 30m indicators for entry timing
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # 30m HMA for short-term trend confirmation
    hma_30m_16 = calculate_hma(close, period=16)
    hma_30m_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for lower TF)
    POSITION_SIZE = 0.20  # Smaller for 30m to reduce fee drag
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_30m_16[i]) or np.isnan(hma_30m_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            signals[i] = 0.0
            continue
        
        # === FILTER 1: 1D PRIMARY TREND DIRECTION ===
        # Only trade WITH the 1d trend (prevents counter-trend losses)
        bull_trend_1d = close[i] > hma_1d_50_aligned[i]
        bear_trend_1d = close[i] < hma_1d_50_aligned[i]
        
        # 1d HMA slope confirmation (21 vs 50)
        bull_slope_1d = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_slope_1d = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === FILTER 2: 4H REGIME (Choppiness Index) ===
        # Only trade when 4h shows trending regime (CHOP < 45)
        trending_regime_4h = chop_4h_aligned[i] < 45.0
        choppy_regime_4h = chop_4h_aligned[i] > 55.0
        
        # === FILTER 3: 4H TREND STRENGTH (ADX) ===
        strong_trend_4h = adx_4h_aligned[i] > 20.0
        
        # === FILTER 4: VOLUME CONFIRMATION ===
        # Volume must be > 1.2x average (confirm conviction)
        volume_spike = volume[i] > 1.2 * vol_avg_20[i]
        
        # === FILTER 5: 30M ENTRY TIMING (Connors RSI Extreme) ===
        # CRSI < 12 = extreme oversold (long entry in bull trend)
        # CRSI > 88 = extreme overbought (short entry in bear trend)
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_extreme_overbought = crsi[i] > 88.0
        
        # === FILTER 6: 30M SHORT-TERM TREND CONFIRMATION ===
        hma_30m_bull = hma_30m_16[i] > hma_30m_48[i]
        hma_30m_bear = hma_30m_16[i] < hma_30m_48[i]
        
        # === ENTRY LOGIC — TRIPLE CONFLUENCE REQUIRED ===
        new_signal = 0.0
        
        # LONG ENTRY: All 6 filters must align
        # 1. 1d bull trend + 2. 4h trending regime + 3. 4h strong trend
        # 4. Volume spike + 5. CRSI extreme oversold + 6. 30m HMA bull
        if (bull_trend_1d and bull_slope_1d and 
            trending_regime_4h and strong_trend_4h and
            volume_spike and crsi_extreme_oversold and hma_30m_bull):
            new_signal = POSITION_SIZE
        
        # SHORT ENTRY: All 6 filters must align
        # 1. 1d bear trend + 2. 4h trending regime + 3. 4h strong trend
        # 4. Volume spike + 5. CRSI extreme overbought + 6. 30m HMA bear
        elif (bear_trend_1d and bear_slope_1d and 
              trending_regime_4h and strong_trend_4h and
              volume_spike and crsi_extreme_overbought and hma_30m_bear):
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip or trend weakness) ===
        # Exit long on 1d trend flip to bear
        if in_position and position_side > 0:
            if bear_trend_1d or (choppy_regime_4h and adx_4h_aligned[i] < 15.0):
                new_signal = 0.0
        
        # Exit short on 1d trend flip to bull
        if in_position and position_side < 0:
            if bull_trend_1d or (choppy_regime_4h and adx_4h_aligned[i] < 15.0):
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