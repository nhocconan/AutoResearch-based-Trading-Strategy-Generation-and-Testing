#!/usr/bin/env python3
"""
Experiment #049: 15m Multi-Timeframe Mean Reversion with 4h HMA Trend Filter
Hypothesis: 15m timeframe captures intraday oscillations while 4h HMA provides trend bias.
Key insight: Mean reversion works better in range markets, but needs HTF trend filter to avoid
catching falling knives. Use RSI(7) for faster signals on 15m, volume confirmation to reduce
false breakouts, and tight ATR stops (2.0x) to limit drawdown.

Why 15m might work: More trade opportunities than 1h/4h/1d, can capture intraday swings.
The 4h HMA filter prevents trading against the major trend. RSI(7) reacts faster than RSI(14).
Volume spike confirmation ensures we're not entering on low-liquidity fakeouts.

Position sizing: 0.25 base, 0.35 for strong confluence. Stoploss at 2.0*ATR (tighter for 15m).
Must generate 10+ trades on train - entry conditions loosened vs failed 15m experiments.
Timeframe: 15m (REQUIRED for exp#049), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_rsi_vol_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / (sma + 1e-10)
    return upper, lower, bb_width

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_volume_zscore(volume, period=20):
    """Calculate volume Z-score for volume spike detection."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_std = pd.Series(volume).rolling(window=period, min_periods=period).std().values
    vol_zscore = (volume - vol_sma) / (vol_std + 1e-10)
    return vol_zscore

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    er = change / (volatility + 1e-10)
    er[:er_period] = np.nan
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, 14)
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    vol_zscore = calculate_volume_zscore(volume, 20)
    kama = calculate_kama(close, 10, 2, 30)
    
    # HMA on 15m for faster trend
    hma_15m = calculate_hma(close, 16)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_9[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        bull_trend_15m = ema_9[i] > ema_21[i] and close[i] > ema_50[i]
        bear_trend_15m = ema_9[i] < ema_21[i] and close[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions (faster for 15m)
        rsi_oversold = rsi_7[i] < 30
        rsi_overbought = rsi_7[i] > 70
        rsi_neutral = 30 <= rsi_7[i] <= 70
        
        # RSI(14) for confirmation
        rsi14_oversold = rsi_14[i] < 35
        rsi14_overbought = rsi_14[i] > 65
        
        # Bollinger Band signals
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.005
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.995
        
        # Z-score filter
        zscore_extreme_long = zscore[i] < -1.5
        zscore_extreme_short = zscore[i] > 1.5
        
        # Volume confirmation
        volume_spike = vol_zscore[i] > 1.0  # Above average volume
        
        # KAMA trend
        kama_bullish = close[i] > kama[i] if not np.isnan(kama[i]) else False
        kama_bearish = close[i] < kama[i] if not np.isnan(kama[i]) else False
        
        # EMA crossover signals
        ema_cross_long = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        ema_cross_short = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # HMA crossover
        hma_cross_long = hma_15m[i] > ema_21[i] and hma_15m[i-1] <= ema_21[i-1]
        hma_cross_short = hma_15m[i] < ema_21[i] and hma_15m[i-1] >= ema_21[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        price_near_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        # BB squeeze (low volatility before breakout)
        bb_squeeze = bb_width[i] < np.nanpercentile(bb_width[:i], 25) if i > 100 else False
        
        new_signal = 0.0
        
        # === BULLISH REGIME (4h HMA + above 200 SMA) ===
        if bull_trend_4h and above_200:
            # Long: Pullback to EMA21 + RSI oversold + volume confirmation
            if price_near_ema21_long and rsi_oversold:
                if volume_spike or rsi14_oversold:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # Long: BB lower band + RSI oversold (mean reversion)
            elif price_at_lower_bb and rsi_oversold:
                if bull_trend_15m or kama_bullish:
                    new_signal = SIZE_BASE
            
            # Long: EMA crossover + volume spike (momentum entry)
            elif ema_cross_long and volume_spike:
                if rsi_neutral and above_200:
                    new_signal = SIZE_HALF
            
            # Long: HMA crossover + trend confirmation
            elif hma_cross_long and bull_trend_15m:
                if volume_spike:
                    new_signal = SIZE_BASE
        
        # === BEARISH REGIME (4h HMA + below 200 SMA) ===
        elif bear_trend_4h and below_200:
            # Short: Pullback to EMA21 + RSI overbought + volume confirmation
            if price_near_ema21_short and rsi_overbought:
                if volume_spike or rsi14_overbought:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
            
            # Short: BB upper band + RSI overbought (mean reversion)
            elif price_at_upper_bb and rsi_overbought:
                if bear_trend_15m or kama_bearish:
                    new_signal = -SIZE_BASE
            
            # Short: EMA crossover + volume spike (momentum entry)
            elif ema_cross_short and volume_spike:
                if rsi_neutral and below_200:
                    new_signal = -SIZE_HALF
            
            # Short: HMA crossover + trend confirmation
            elif hma_cross_short and bear_trend_15m:
                if volume_spike:
                    new_signal = -SIZE_BASE
        
        # === TRANSITION/NEUTRAL REGIME ===
        else:
            # Conservative mean reversion only
            if price_at_lower_bb and rsi_oversold and zscore_extreme_long:
                new_signal = SIZE_HALF
            elif price_at_upper_bb and rsi_overbought and zscore_extreme_short:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals