#!/usr/bin/env python3
"""
Experiment #008: 30m Funding Rate Contrarian + 4h HMA Trend Bias
Hypothesis: Funding rate mean reversion (proven edge for BTC/ETH) combined with 4h trend bias 
will outperform pure trend strategies in bear/range markets. Key insight from research: 
funding rate z-score < -2 → long, > +2 → short has Sharpe 0.8-1.5 through 2022 crash.
Adding 4h HMA for regime filter prevents counter-trend trades in strong trends.
Volatility filter (ATR ratio) avoids entries during panic spikes.
Timeframe: 30m (REQUIRED for exp#008), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.15 half - discrete levels to minimize fee churn.
Why this might work: Funding rates capture crowded positioning, 4h HMA filters regime,
volatility filter avoids panic entries. Should generate 20-40 trades/year.
Must generate 10+ trades on train, 3+ on test - conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_funding_contrarian_4h_hma_v1"
timeframe = "30m"
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

def calculate_zscore(series, period=30):
    """Calculate rolling z-score."""
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(series).rolling(window=period, min_periods=period).std().values
    zscore = (series - sma) / (std + 1e-10)
    return zscore

def load_funding_data(symbol):
    """Load funding rate data from processed parquet files."""
    import os
    symbol_map = {
        'BTCUSDT': 'BTC',
        'ETHUSDT': 'ETH',
        'SOLUSDT': 'SOL'
    }
    base_symbol = symbol_map.get(symbol, symbol.replace('USDT', ''))
    funding_path = f"data/processed/funding/{base_symbol}.parquet"
    
    if os.path.exists(funding_path):
        df_funding = pd.read_parquet(funding_path)
        return df_funding['funding_rate'].values
    else:
        # Fallback: use price-based proxy (returns = 0 means no funding signal)
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Get symbol for funding data
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if 'symbol' in prices.columns else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Load funding rate data (contrarian signal)
    funding_rates = load_funding_data(symbol)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_zscore(funding_rates[:n], 30)
    else:
        funding_zscore = np.zeros(n)  # No funding signal fallback
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    # ATR ratio for volatility regime
    atr_ratio = atr / (atr_30 + 1e-10)
    
    # Bollinger Bands for mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    bb_width = (bb_upper - bb_lower) / (sma_20 + 1e-10)
    
    # Price position within BB
    bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # Funding rate contrarian signal (BEST edge for BTC/ETH)
        funding_extreme_long = funding_zscore[i] < -1.5  # Very negative funding → long
        funding_extreme_short = funding_zscore[i] > 1.5  # Very positive funding → short
        funding_moderate_long = funding_zscore[i] < -0.5
        funding_moderate_short = funding_zscore[i] > 0.5
        
        # Volatility filter - avoid panic entries
        vol_normal = atr_ratio[i] < 1.5  # ATR not spiking
        vol_low = atr_ratio[i] < 1.0  # Low vol = good for entries
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 < rsi[i] < 65
        
        # Bollinger mean reversion
        price_at_bb_lower = bb_position[i] < 0.1  # Near lower band
        price_at_bb_upper = bb_position[i] > 0.9  # Near upper band
        
        # Volume confirmation
        avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_above_avg = volume[i] > avg_vol[i] * 0.8 if not np.isnan(avg_vol[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Funding contrarian + 4h bullish trend
        if funding_extreme_long and bull_trend_4h and vol_normal:
            new_signal = SIZE_BASE
        
        # Secondary: Funding moderate + RSI oversold + trend support
        elif funding_moderate_long and rsi_oversold and (bull_trend_4h or above_200):
            new_signal = SIZE_HALF
        
        # Tertiary: BB mean reversion in bullish regime
        elif price_at_bb_lower and rsi_oversold and bull_trend_4h and vol_low:
            new_signal = SIZE_HALF
        
        # Momentum: RSI bounce with trend
        elif rsi[i] > 40 and rsi[i-1] < 35 and bull_trend_4h and vol_normal:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        # Primary: Funding contrarian + 4h bearish trend
        if funding_extreme_short and bear_trend_4h and vol_normal:
            new_signal = -SIZE_BASE
        
        # Secondary: Funding moderate + RSI overbought + trend resistance
        elif funding_moderate_short and rsi_overbought and (bear_trend_4h or below_200):
            new_signal = -SIZE_HALF
        
        # Tertiary: BB mean reversion in bearish regime
        elif price_at_bb_upper and rsi_overbought and bear_trend_4h and vol_low:
            new_signal = -SIZE_HALF
        
        # Momentum: RSI rejection with trend
        elif rsi[i] < 60 and rsi[i-1] > 65 and bear_trend_4h and vol_normal:
            new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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