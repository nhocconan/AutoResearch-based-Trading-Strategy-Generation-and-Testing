#!/usr/bin/env python3
"""
Experiment #452: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Funding Contrarian + Dual Regime

Hypothesis: Building on #442 (Sharpe=0.185 on 12h) and current best (Sharpe=0.612 on 4h),
switch to 12h primary with simpler, more robust signals. Key improvements:
1. KAMA (Kaufman Adaptive) instead of HMA - better noise filtering in crypto
2. Funding Rate contrarian signal (proven edge for BTC/ETH in bear/range markets)
3. Simplified regime: ADX for trend strength + CHOP for choppy detection
4. Relaxed entry conditions to ensure 30+ trades on train (avoid 0-trade failure)
5. 1d + 1w KAMA confluence for stronger trend bias
6. ATR trailing stop (2.5x) + CRSI extreme exit for take profit
7. Position size: 0.25 base, 0.30 on strong confluence, discrete levels

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -35%
Timeframe: 12h (proven best for swing trading, fewer trades = less fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_funding_regime_crsi_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i - er_period])
        volatility = np.nansum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 1e-10:
            er[i] = change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_di_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_di_s / (atr + 1e-10)
        minus_di = 100.0 * minus_di_s / (atr + 1e-10)
    
    dx = np.zeros(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def load_funding_rate(prices):
    """Load funding rate data for contrarian signal."""
    try:
        import os
        symbol = "BTCUSDT"  # Default, will try to match
        if "ETH" in str(prices.columns):
            symbol = "ETHUSDT"
        elif "SOL" in str(prices.columns):
            symbol = "SOLUSDT"
        
        # Try to find funding data
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return df_funding
        return None
    except:
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Recalculate KAMA with different periods
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=21)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF KAMA for bias (1d and 1w)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    # Load funding rate for contrarian signal
    funding_df = load_funding_rate(prices)
    funding_zscore = np.zeros(n)
    if funding_df is not None and 'funding_rate' in funding_df.columns:
        try:
            funding_rates = funding_df['funding_rate'].values
            if len(funding_rates) >= 30:
                funding_mean = np.nanmean(funding_rates[-30:])
                funding_std = np.nanstd(funding_rates[-30:])
                if funding_std > 1e-10:
                    funding_zscore = (funding_rates - funding_mean) / funding_std
        except:
            pass
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(adx[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index + ADX) ===
        regime_chop = chop[i] > 55.0  # Range market (relaxed from 61.8)
        regime_trend = chop[i] < 45.0 and adx[i] > 20.0  # Trending market
        
        # === HTF TREND BIAS (1d + 1w KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        htf_bullish = price_above_kama_1d and price_above_kama_1w
        htf_bearish = price_below_kama_1d and price_below_kama_1w
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === PRIMARY TREND (12h KAMA) ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED FOR TRADES ===
        crsi_oversold = crsi[i] < 35.0  # Relaxed from 30
        crsi_overbought = crsi[i] > 65.0  # Relaxed from 70
        crsi_extreme_oversold = crsi[i] < 20.0
        crsi_extreme_overbought = crsi[i] > 80.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND SIGNALS ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 22.0  # Relaxed from 25
        adx_weak = adx[i] < 18.0
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.52  # Relaxed from 0.55
        volume_bearish = taker_ratio[i] < 0.48  # Relaxed from 0.45
        
        # === FUNDING RATE CONTRARIAN ===
        funding_extreme_short = funding_zscore[i] > 2.0  # High funding = short contrarian
        funding_extreme_long = funding_zscore[i] < -2.0  # Low funding = long contrarian
        
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
        
        # === REGIME 1: CHOPPY/RANGE — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold OR BB oversold + HTF not bearish
            if (crsi_oversold or bb_oversold) and not htf_bearish:
                signal_strength = 1
                if crsi_extreme_oversold or bb_oversold:
                    signal_strength = 2
                if funding_extreme_long:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
            
            # Short: CRSI overbought OR BB overbought + HTF not bullish
            if (crsi_overbought or bb_overbought) and not htf_bullish:
                if desired_signal == 0:
                    signal_strength = 1
                    if crsi_extreme_overbought or bb_overbought:
                        signal_strength = 2
                    if funding_extreme_short:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
        
        # === REGIME 2: TRENDING — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout OR KAMA bullish + HTF bullish
            if (donchian_breakout_long or (kama_bullish and adx_strong)):
                signal_strength = 1
                if htf_bullish:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                if kama_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
            
            # Short: Donchian breakdown OR KAMA bearish + HTF bearish
            if (donchian_breakout_short or (kama_bearish and adx_strong)):
                if desired_signal == 0:
                    signal_strength = 1
                    if htf_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    if kama_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
        
        # === REGIME 3: TRANSITION — HYBRID ===
        else:
            # KAMA crossover with HTF confirmation
            if kama_bullish and not htf_bearish:
                desired_signal = position_size * 0.5
            elif kama_bearish and not htf_bullish:
                desired_signal = -position_size * 0.5
            elif crsi_extreme_oversold and not htf_bearish:
                desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and not htf_bullish:
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
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (kama_bullish or price_above_kama_1d):
                desired_signal = position_size
            elif position_side < 0 and (kama_bearish or price_below_kama_1d):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
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