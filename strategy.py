#!/usr/bin/env python3
"""
Experiment #453: 1d Primary + 1w HTF — Regime-Adaptive with Relaxed Entries

Hypothesis: After #451 (4h, Sharpe=-0.080) and #452 (12h, Sharpe=0.209), shift to 
1d primary timeframe which has proven more stable for crypto swing trading. Key changes:
1. 1d primary with 1w HMA for major trend bias (reduces whipsaws vs 4h)
2. Connors RSI with RELAXED thresholds (25/75 not 20/80) to ensure >=30 trades/year
3. Choppiness Index regime switch (CHOP>61.8=range MR, <38.2=trend TF)
4. Fisher Transform for precise reversal timing in bear markets
5. Donchian(20) breakout for trend entries with 1w confirmation
6. ATR(14) trailing stop at 2.5x + CRSI extreme exit (80/20)
7. Position size: 0.25 base, 0.30 on strong confluence, discrete levels
8. Volume confirmation via taker buy ratio (>0.55 long, <0.45 short)

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -35%
Timeframe: 1d (proven for swing trading, fewer fees than lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_fisher_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(close[i-period+1:i+1])
        lowest = np.nanmin(close[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
            continue
        
        w = 0.66 * ((close[i] - lowest) / range_val - 0.5) + 0.67 * fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
        w = np.clip(w, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + w) / (1.0 - w))
        trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, trigger

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    rsi = calculate_rsi(close, rsi_period)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_di_s / (atr + 1e-10)
        minus_di = 100.0 * minus_di_s / (atr + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.nansum(np.abs(np.diff(close[i-er_period:i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    kama_20 = calculate_kama(close, er_period=10)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF HMA for bias (1w)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size for 1d
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(adx[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market
        regime_trend = chop[i] < 38.2  # Trending market
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        htf_bullish = price_above_hma_1w
        htf_bearish = price_below_hma_1w
        
        # === PRIMARY TREND (1d HMA + KAMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        kama_bullish = close[i] > kama_20[i]
        kama_bearish = close[i] < kama_20[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > fisher_trigger[i] and fisher_trigger[i] < -0.5
        fisher_short = fisher[i] < fisher_trigger[i] and fisher_trigger[i] > 0.5
        fisher_extreme_long = fisher[i] < -1.2
        fisher_extreme_short = fisher[i] > 1.2
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED for more trades ===
        crsi_oversold = crsi[i] < 28.0  # Relaxed from 20
        crsi_overbought = crsi[i] > 72.0  # Relaxed from 80
        crsi_extreme_oversold = crsi[i] < 18.0
        crsi_extreme_overbought = crsi[i] > 82.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 22.0  # Relaxed from 25
        adx_weak = adx[i] < 18.0
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.52  # Relaxed from 0.55
        volume_bearish = taker_ratio[i] < 0.48  # Relaxed from 0.45
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold OR Fisher extreme + HTF not bearish
            if crsi_oversold and not htf_bearish:
                signal_strength = 1
                if crsi_extreme_oversold:
                    signal_strength = 2
                if fisher_extreme_long:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                if kama_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
            
            # Short: CRSI overbought OR Fisher extreme + HTF not bullish
            if crsi_overbought and not htf_bullish:
                if desired_signal == 0 or abs(desired_signal) < position_size * 0.5:
                    signal_strength = 1
                    if crsi_extreme_overbought:
                        signal_strength = 2
                    if fisher_extreme_short:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    if kama_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout OR Fisher crossover + HTF bullish
            if donchian_breakout_long and adx_strong:
                signal_strength = 1
                if htf_bullish:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                if hma_bullish:
                    signal_strength += 1
                if kama_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
            elif fisher_long and htf_bullish:
                if desired_signal == 0:
                    signal_strength = 1
                    if hma_bullish:
                        signal_strength += 1
                    if volume_bullish:
                        signal_strength += 1
                    desired_signal = position_size * 0.7 * (0.7 + 0.3 * signal_strength / 3)
            
            # Short: Donchian breakdown OR Fisher crossover + HTF bearish
            if donchian_breakout_short and adx_strong:
                if desired_signal == 0 or abs(desired_signal) < position_size * 0.5:
                    signal_strength = 1
                    if htf_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    if hma_bearish:
                        signal_strength += 1
                    if kama_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
            elif fisher_short and htf_bearish:
                if desired_signal == 0:
                    signal_strength = 1
                    if hma_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * 0.7 * (0.7 + 0.3 * signal_strength / 3)
        
        # === REGIME 3: TRANSITION (38.2-61.8) — HYBRID ===
        else:
            # Fisher reversals with HTF confirmation (more lenient)
            if fisher_extreme_long and not htf_bearish:
                desired_signal = position_size * 0.6
            elif fisher_extreme_short and not htf_bullish:
                if desired_signal == 0:
                    desired_signal = -position_size * 0.6
            elif crsi_extreme_oversold and not htf_bearish:
                if desired_signal == 0:
                    desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and not htf_bullish:
                if desired_signal == 0:
                    desired_signal = -position_size * 0.5
            elif donchian_breakout_long and htf_bullish:
                if desired_signal == 0:
                    desired_signal = position_size * 0.5
            elif donchian_breakout_short and htf_bearish:
                if desired_signal == 0:
                    desired_signal = -position_size * 0.5
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 82.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 18.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_1w):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_1w):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.20
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.20
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals